import json
import os
import pickle
from uuid import uuid4

import torch
from PIL import Image
from torchvision import transforms
from doctr.io import DocumentFile
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from celery import Celery

import table_assembly
from net import Net
from orientation_net import OrientationNet

fastapi_app = FastAPI()

REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL")
REDIS_BACKEND_URL = os.getenv("REDIS_BACKEND_URL")
ALLOWED_EXTENSION = os.getenv("ALLOWED_EXTENSION").split()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

celery_app = Celery("worker", broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL)
celery_app.conf.update({"worker_hijack_root_logger": False})


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


@fastapi_app.post("/upload")
def upload(file: UploadFile):
    if not file.file:
        raise HTTPException(status_code=400, detail="No file provided")

    if "." not in file.filename:
        raise HTTPException(status_code=400, detail="No file extension found")

    file_name, file_ext = file.filename.rsplit(".", maxsplit=1)

    if file_ext.lower() not in ALLOWED_EXTENSION:
        raise HTTPException(status_code=400, detail="Incorrect file extension")

    unique_filename = f"{uuid4()}.{file_ext}"
    save_path = os.path.join(os.path.dirname(__file__), "img", unique_filename)

    with open(save_path, "wb") as file_object:
        file_object.write(file.file.read())

    orientation = orient_classifier.predict(obtaining_embedding(save_path))[0]
    with Image.open(save_path) as img:
        corrected_img = rotate_image(img, orientation)
        corrected_img.save(save_path)

    if not classifier.predict(obtaining_embedding(save_path))[0]:
        raise HTTPException(status_code=400, detail="Incorrect file type")

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

        return df.to_json(orient="records", force_ascii=False)

    except Exception as exc:
        raise exc
