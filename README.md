# DistTrainBench prototype

초기 개발용 CPU-only Docker 구성이다. coordinator 1개와 worker 2개가 같은 이미지와 bridge network를 사용한다.

## Base image

- Python 3.12
- CPU-only PyTorch 2.11.0 + torchvision 0.26.0
- Pillow, NumPy, Requests
- bash, curl, git

나머지 Python 패키지는 에이전트가 설치하도록 둔다. Codex/Claude CLI 설치와 worker 제어 인터페이스는 다음 단계에서 추가한다.

## Start

```bash
cp .env.example .env
docker compose up --build -d
```

The default development quotas are 1 CPU and 4 GB memory for the coordinator, and 1 CPU and 4 GB for each worker. Change them in `.env` after checking the host's available resources. The three containers share `/workspace`, `/data`, and `/runs`; workers resolve on the `bench` network as `worker-1` and `worker-2`.

The API key variables, when set, are passed to the coordinator only. They are not baked into the image and are not passed to workers.

Useful commands:

```bash
docker compose exec coordinator bash
docker compose exec worker-1 bash
docker compose exec worker-2 bash
docker compose down
```
