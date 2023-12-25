FROM python:3.10-slim

WORKDIR /app

COPY /requirements.txt /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install \
    --default-timeout=100 \
    torch==2.0.1 \
    torchvision==0.15.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

COPY /app /app
