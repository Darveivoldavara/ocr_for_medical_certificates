FROM python:3.10.6

COPY /opt /app
WORKDIR /app

RUN pip install --default-timeout=100 future -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]