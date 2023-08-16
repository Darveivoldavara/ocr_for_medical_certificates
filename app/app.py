import os
import logging
import pickle
from typing import List

from doctr.io import DocumentFile
from celery import Celery
from celery.signals import setup_logging
from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse

import table_assembly


app = FastAPI()
celery_app = Celery("worker", broker="redis://redis:6379/0",
                    backend="redis://redis")
celery_app.conf.update({'worker_hijack_root_logger': False})


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("logs.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


try:
    model = pickle.load(
        open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb"))
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error while loading model: {e}")


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


@app.post("/ocr")
def process_request(file: UploadFile):
    task = process_file.delay(file.file.read())
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
def process_file(file_content: bytes):
    try:
        save_pth = os.path.join(os.path.dirname(__file__), "img", "img.jpg")
        with open(save_pth, "wb") as fid:
            fid.write(file_content)

        image = DocumentFile.from_images(f"{save_pth}")
        result = model(image)
        jsn = result.pages[0].export()
        lst = []
        for b in jsn["blocks"]:
            for l in b["lines"]:
                for w in l["words"]:
                    lst.append(w["value"])

        df = table_assembly.assembly(lst)

        logging.info(
            f'Result: {df.to_json(orient="records", force_ascii=False)}')

        return df.to_json(orient="records",
                          force_ascii=False)
    except Exception as e:
        logging.error(f"Error while processing file: {e}")
        raise e
