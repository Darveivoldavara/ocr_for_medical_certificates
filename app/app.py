import table_assembly
from doctr.io import DocumentFile
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import argparse
import os
import pickle

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "OK"}


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
    save_pth = os.path.join(os.path.dirname(__file__), "img", 'img.jpg')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read()) 

    
    image = DocumentFile.from_images(f'{save_pth}')
    result = model(image)
    jsn = result.pages[0].export()
    lst = []
    for b in jsn['blocks']:
        for l in b['lines']:
            for w in l['words']:
                lst.append(w['value'])

    df = table_assembly.assembly(lst)
    
    return df.to_json(orient="records",
                      force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))

    uvicorn.run(app, **args)


