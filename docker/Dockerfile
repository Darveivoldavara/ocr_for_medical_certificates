FROM python:3.12-slim

WORKDIR /app

COPY /requirements.txt /app
COPY /third_party_licenses.md /app

RUN apt-get update && apt-get install ffmpeg -y && rm -rf /var/lib/apt/lists/*
RUN pip install \
    --default-timeout=100 \
    torch==2.2.0 \
    torchvision==0.17.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

COPY /app /app
