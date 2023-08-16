docker build -t docker_redis -f Dockerfile.redis ../
docker build -t docker_redis -f Dockerfile.redis ../
docker build -t docker_worker -f Dockerfile.worker ../
docker compose up