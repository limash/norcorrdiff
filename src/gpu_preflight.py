#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU availability preflight check for Vertex AI training."""

import ctypes
import sys

import torch


def main():
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")

    try:
        ctypes.CDLL("libcuda.so.1")
        print("libcuda_load=ok")
    except OSError as error:
        print(f"libcuda_load=failed: {error}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available inside the container.")
        sys.exit(1)

    print("GPU preflight OK, starting training...")


if __name__ == "__main__":
    main()
