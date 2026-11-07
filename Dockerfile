FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        bash \
        ca-certificates \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pin the CPU wheel pair so the coordinator and workers start from the same
# PyTorch runtime. Install general-purpose Python packages from PyPI separately.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 \
        torchvision==0.26.0 \
    && python -m pip install --no-cache-dir Pillow numpy requests

CMD ["sleep", "infinity"]
