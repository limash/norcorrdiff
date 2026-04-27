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

RUN python3.11 -m venv $VIRTUAL_ENV \
 && $VIRTUAL_ENV/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt .

# 1) Pin Torch to a CUDA 12.1 build that is broadly compatible with managed T4 hosts
RUN pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) Install matching PyG wheels
RUN pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 3) Install PhysicsNeMo and your packages
RUN pip install \
    "nvidia-physicsnemo[cu12,utils-extras,mesh-extras,datapipes-extras,gnns] @ https://github.com/NVIDIA/physicsnemo/archive/ccbf9a07b7e1b8cf926e638713df92639945a7ee.tar.gz" \
    "tensorboard>=2.16" \
    -r requirements.txt

COPY src ./src

CMD ["python", "src/train.py"]

