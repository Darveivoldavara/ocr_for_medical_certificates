docker build -t darveivoldavara/ocr_for_medical_certificates:redis -f Dockerfile.redis ../
docker build -t darveivoldavara/ocr_for_medical_certificates:web -f Dockerfile.web ../
docker build -t darveivoldavara/ocr_for_medical_certificates:worker -f Dockerfile.worker ../
docker compose up