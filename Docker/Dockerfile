FROM python:3.10.4

COPY /app /app
WORKDIR /app

RUN pip install --default-timeout=100 -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

CMD ["python", "app.py"]