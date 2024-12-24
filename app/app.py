from datetime import datetime
import os
import logging
import pickle
import json
from uuid import uuid4

from PIL import Image
import torch
from torchvision import transforms

from doctr.io import DocumentFile
from celery import Celery
from celery.signals import setup_logging
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse

import table_assembly
from net import Net
from orientation_net import OrientationNet

fastapi_app = FastAPI()

REDIS_BROKER_URL = os.environ.get("REDIS_BROKER_URL")
REDIS_BACKEND_URL = os.environ.get("REDIS_BACKEND_URL")
ALLOWED_EXTENSION = os.getenv("ALLOWED_EXTENSION").split()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

celery_app = Celery("worker", broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)
celery_app.conf.update({"worker_hijack_root_logger": False})


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_model(file_name):
    with open(os.path.join(MODEL_PATH, file_name), "rb") as file:
        return pickle.load(file)


try:
    model = load_model("model.pkl")
    encoder = load_model("beit_encoder.pkl")
    classifier = load_model("skorch_ffnn_classifier.pkl")
    orient_classifier = load_model("orientation_classifier.pkl")
except (FileNotFoundError, IOError) as e:
    raise e


def obtaining_embedding(img_path):
    preprocess = transforms.Compose(
        [
            transforms.Resize(384),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = encoder(image)
        embedding = output[0][:, 0, :]
    return embedding


def rotate_image(image, orientation):
    if orientation == 1:
        return image.rotate(90, expand=True)
    elif orientation == 2:
        return image.rotate(-90, expand=True)
    elif orientation == 3:
        return image.rotate(180, expand=True)
    return image

def clean_json(input_json: str) -> str:
    """
    Исправляет JSON-строку согласно заданным правилам.
    
    Args:
        input_json (str): Строка с исходным JSON.
        
    Returns:
        str: Строка с исправленным JSON.
    """
    
    def fix_date(date_str: str) -> str:
        """Исправляет дату в формате MM.DD.YYYY на формат YYYY-MM-DD."""
        try:
            # Определяем формат и преобразуем дату
            if "." in date_str:
                return datetime.strptime(date_str, "%m.%d.%Y").strftime("%Y-%m-%d")
            else:
                return date_str
        except:
            return None  # Возвращаем None, если формат некорректный
        
    valid_blood_classes = {"platelets", "blood", "plasma"}
    valid_payment_types = {"free", "payed"}


    data = json.loads(input_json)

    fixed_data = []
    for record in data:
        blood_class = record["blood_class"]
        if blood_class not in valid_blood_classes:
            record["blood_class"] = "blood"
        
        payment_type = record["payment_type"]
        if payment_type not in valid_payment_types:
            record["payment_type"] = "free"
        
        donate_at = fix_date(record["donate_at"])
        if donate_at is None:
            continue  # Удаляем запись, если дата некорректна
        record["donate_at"] = donate_at
        
        fixed_data.append(record)

    return json.dumps(fixed_data, ensure_ascii=False)

@fastapi_app.get("/")
def main():
    html_content = """
            <body>
            <form action="/upload" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
            </body>
            """
    return HTMLResponse(content=html_content)


@fastapi_app.get("/health")
def health():
    return {"status": "OK"}


@fastapi_app.post("/upload")
def upload(file: UploadFile):
    if not file.file:
        raise HTTPException(status_code=400, detail="No file provided")

    if "." not in file.filename:
        raise HTTPException(
            status_code=400, detail="No file extension found. Check file name"
        )

    file_ext = file.filename.rsplit(".", maxsplit=1)[1]

    if file_ext.lower() not in ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=400,
            detail="Incorrect file extension. Allowed formats are: {ALLOWED_EXTENSION}",
        )
    unique_filename = f"{uuid4()}.{file_ext}"
    save_path = os.path.join(os.path.dirname(__file__), "img", unique_filename)

    with open(save_path, "wb") as file_object:
        file_object.write(file.file.read())

    orientation = orient_classifier.predict(obtaining_embedding(save_path))[0]
    with Image.open(save_path) as img:
        corrected_img = rotate_image(img, orientation)
        corrected_img.save(save_path)

    if not classifier.predict(obtaining_embedding(save_path))[0]:
        raise HTTPException(
            status_code=400,
            detail="Incorrect file type. The service only works with certificate of form 405",
        )
    task = process_file.delay(save_path)
    return {"task_id": task.id}


@fastapi_app.get("/result/{task_id}")
def get_result(task_id: str):
    task = celery_app.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "success", "result": json.loads(task.result)}
    else:
        return {"status": task.state}


@celery_app.task(name="process_file")
def process_file(file_path: str):
    try:
        image = DocumentFile.from_images(file_path)
        result = model(image)
        jsn = result.pages[0].export()
        lst = [
            w["value"] for b in jsn["blocks"] for l in b["lines"] for w in l["words"]
        ]
        df = table_assembly.assembly(lst)

        if os.path.exists(file_path):
            os.remove(file_path)

        json_ = df.to_json(orient="records", force_ascii=False)
        json_ = clean_json(json_)
        return json_

    except Exception as exc:
        raise exc
