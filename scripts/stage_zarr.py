"""Stage a gs:// Zarr store to a local directory, in parallel.

`cp -r` over the /gcs FUSE mount is single-threaded, and the cut store is tens
of thousands of tiny per-sample chunk files (chunks=(1, C, 128, 128); see
make_cut_zarr.py). A serial copy opens each file with its own network
round-trip, so it is latency-bound and takes hours. gcsfs lists the objects and
downloads them concurrently straight from gs://, bypassing the FUSE mount.

Example:
    python scripts/stage_zarr.py \
        --src gs://norcorrdiff-us/taiwan_dataset/cwa_dataset/cwa_dataset_cut.zarr \
        --dst /workspace/data_local/cwa_dataset_cut.zarr
"""

import argparse
import logging
import os
import time

import gcsfs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("stage_zarr")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source gs:// Zarr store.")
    p.add_argument("--dst", required=True, help="Local destination directory.")
    args = p.parse_args()

    if not args.src.startswith("gs://"):
        raise ValueError(f"--src must be a gs:// path, got {args.src}")

    fs = gcsfs.GCSFileSystem()
    base = args.src[len("gs://"):].rstrip("/")  # 'bucket/path/store.zarr'
    rpaths = fs.find(base)  # ['bucket/path/store.zarr/cwb/0.0.0.0', ...]
    if not rpaths:
        raise FileNotFoundError(f"No objects found under {args.src}")

    # The store is zarr v2 (written by make_cut_zarr.py, which pins zarr<3), but
    # the training container runs zarr v3. A stray v3 `zarr.json` alongside the
    # v2 metadata makes v3 pick v3 mode and then fail to find v3 consolidated
    # metadata. Drop any zarr.json so the local copy is unambiguously v2, which
    # zarr v3 reads via the v2 `.zmetadata`.
    n_before = len(rpaths)
    rpaths = [r for r in rpaths if os.path.basename(r) != "zarr.json"]
    if len(rpaths) != n_before:
        logger.info("Skipping %d stray v3 zarr.json object(s)", n_before - len(rpaths))
    lpaths = [os.path.join(args.dst, os.path.relpath(r, base)) for r in rpaths]

    for d in {os.path.dirname(p) for p in lpaths}:
        os.makedirs(d, exist_ok=True)

    logger.info("Staging %d objects %s -> %s", len(rpaths), args.src, args.dst)
    t0 = time.time()
    fs.get(rpaths, lpaths)  # async filesystem: transfers run concurrently
    logger.info("Staged %d objects in %.1f min", len(rpaths), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
