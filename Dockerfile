FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        bash \
        ca-certificates \
        coreutils \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Node.js is required by the Codex CLI and Claude Code packages.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install --no-install-recommends -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY disttrainbench /opt/disttrainbench/disttrainbench
ENV PYTHONPATH=/opt/disttrainbench

# Pin the current stable CPU wheel pair so the coordinator and workers start from the same
# PyTorch runtime. Install general-purpose Python packages from PyPI separately.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.15.0 \
        torchvision==0.29.2 \
    && python -m pip install --no-cache-dir Pillow numpy requests \
    && npm install --global --no-fund --no-audit \
        @openai/codex@0.144.0 \
        @anthropic-ai/claude-code@2.1.219

CMD ["sleep", "infinity"]
