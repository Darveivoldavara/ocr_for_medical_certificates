import os
import logging
import pickle
from typing import List

from PIL import Image
from torch import no_grad
from torchvision import transforms

from doctr.io import DocumentFile
from celery import Celery
from celery.signals import setup_logging
from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse

import table_assembly


app = FastAPI()
celery_app = Celery("worker", broker="redis://redis:6379/0",
                    backend="redis://redis")
celery_app.conf.update({"worker_hijack_root_logger": False})
allowed_extensions = ["jpg", "jpeg", "png", "raw", "psd", "bmp"]


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


try:
    model = pickle.load(open(
        os.path.join(os.path.dirname(__file__), "model.pkl"), "rb"
    ))
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error while loading model: {e}")

try:
    encoder = pickle.load(open(
        os.path.join(os.path.dirname(__file__), "beit_encoder.pkl"), "rb"
    ))
    logging.info("Encoder loaded successfully")
except Exception as e:
    logging.error(f"Error while loading encoder: {e}")

try:
    classifier = pickle.load(open(
        os.path.join(os.path.dirname(__file__), "skorch_ffnn_classifier.pkl"),
        "rb"
    ))
    logging.info("Classifier loaded successfully")
except Exception as e:
    logging.error(f"Error while loading classifier: {e}")


def obtaining_embedding(img_path):
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    image = Image.open(img_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with no_grad():
        output = encoder(image)
        embedding = output[0][:, 0, :]
    return embedding


@app.get("/")
def main():
    html_content = """
            <body>
            <form action="/ocr" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
            </body>
            """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/ocr")
def process_request(file: UploadFile):
    file_name, file_ext = file.filename.rsplit(".", maxsplit=1)
    if file_ext not in allowed_extensions:
        logging.error(
            f'{file.filename} has an unsupported format. Allowed formats are: {", ".join(allowed_extensions)}')
        return f"Wrong file format. Allowed formats are: {', '.join(allowed_extensions)}"
    logging.info(f"Received file: {file.filename}")

    save_path = os.path.join(os.path.dirname(__file__), "img", file.filename)
    with open(save_path, "wb") as fid:
        fid.write(file.file.read())
    if not classifier.predict(obtaining_embedding(save_path))[0]:
        logging.error(
            f"The uploaded image is not a medical certificate of form 405. The service only works with them")
        return f"The uploaded image is not a medical certificate of form 405. The service only works with them"

    task = process_file.delay(save_path, file_name)
    logging.info(f"Task {task.id} sent to Celery")
    return {"task_id": task.id}


@app.get("/result/{task_id}")
def get_result(task_id: str):
    task = celery_app.AsyncResult(task_id)
    if task.state == "SUCCESS":
        logging.info(f"Task {task_id} completed successfully")
        return task.result
    else:
        logging.warning(
            f"Task {task_id} is not completed yet, current state: {task.state}")
        return {"status": task.state}


@celery_app.task(name="process_file")
def process_file(file_path: str, file_name: str):
    try:
        image = DocumentFile.from_images(file_path)
        result = model(image)
        jsn = result.pages[0].export()
        lst = []
        for b in jsn["blocks"]:
            for l in b["lines"]:
                for w in l["words"]:
                    lst.append(w["value"])

        df = table_assembly.assembly(lst)
        df[[
            "blood_station_id",
            "plan_date",
            "email",
            "is_out",
            "volume",
            "payment_cost",
            "city_id",
            "first_name",
            "middle_name",
            "last_name"
        ]] = ""
        df["image_id"] = file_name
        df["with_image"] = True

        logging.info(
            f'Result: {df.to_json(orient="records", force_ascii=False)}')

        return df.to_json(orient="records",
                          force_ascii=False)
    except Exception as e:
        logging.error(f"Error while processing file: {e}")
        raise e
