FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    build-essential \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
 && rm -rf /var/lib/apt/lists/*

# Cloud Storage FUSE: lets the Vertex job self-mount the GCS bucket with file/metadata
# caching (the auto-mounted /gcs path is uncached and is the training data bottleneck).
RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-jammy main" \
      > /etc/apt/sources.list.d/gcsfuse.list \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && apt-get update && apt-get install -y --no-install-recommends gcsfuse \
 && gcsfuse --version \
 && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv $VIRTUAL_ENV \
 && $VIRTUAL_ENV/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# 1) Pin Torch 2.6.x (published on cu124 wheels) while keeping base image CUDA 12.1
RUN pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 2) Install matching PyG wheels
RUN pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_geometric \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 3) Install PhysicsNeMo and your packages
COPY requirements.txt .
RUN pip install \
    "nvidia-physicsnemo[cu12,utils-extras,mesh-extras,datapipes-extras,gnns] @ https://github.com/NVIDIA/physicsnemo/archive/refs/tags/v2.0.0.tar.gz" \
    "tensorboard>=2.16" \
    -r requirements.txt

COPY src ./src

CMD ["python", "src/train.py"]

