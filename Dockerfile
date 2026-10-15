# Reproducible benchmark image. Build:  docker build -t inferno .
# Run:  docker run --gpus all --rm -v "$PWD/results:/workspace/results" inferno make smoke
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make curl ca-certificates openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
RUN uv venv --python 3.12 /opt/venv && . /opt/venv/bin/activate && uv pip install -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv

COPY . .
RUN uv pip install -e .

CMD ["make", "test"]
