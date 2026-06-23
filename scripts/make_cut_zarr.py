# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a spatially-cropped ("cut") copy of the CWB Zarr store.

The source store (cwa_dataset.zarr) chunks every variable as one full timestep
(all channels, full 450x450, uncompressed). The Taiwan configs only ever train
on a fixed snippet window, so each sample reads ~19 MB from GCS to keep ~1 MB.
This tool writes a derived store that contains only the snippet region, chunked
one-sample-per-chunk, so reads match what training consumes.

It streams timestep-by-timestep directly gs:// -> gs:// (one ~19 MB chunk in
RAM at a time, parallelized across threads) so nothing is staged on local disk.

What is changed vs. the source:
  - `cwb`, `era5`, `XLAT`, `XLONG` are cropped to the snippet window.
  - Every other array (valid masks, time, *_center/_scale/_variable/_pressure,
    staggered grids) is copied verbatim, so valid-time indexing and the
    normalization statistics are bit-identical.
  - Values stay RAW (un-normalized); the dataset normalizes on read.

After running, point the `cwb` dataset config at the new store and set
`img_shape_x: 128`, `img_shape_y: 128`, `add_grid: false`, `ds_factor: 1`
(in_channels/out_channels unchanged).

Run it inside GCP (a VM/Vertex job in the bucket's region) to avoid egress
cost and get fast bucket<->bucket throughput. Requires Application Default
Credentials: `gcloud auth application-default login`.

Example:
    python scripts/make_cut_zarr.py \
        --src gs://norcorrdiff-us/taiwan_dataset/cwa_dataset/cwa_dataset.zarr \
        --dst gs://norcorrdiff-us/taiwan_dataset/cwa_dataset/cwa_dataset_cut.zarr \
        --x-offset 250 --y-offset 50 --length 128 --workers 32
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numpy as np
import zarr
from numcodecs import Blosc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("make_cut_zarr")

# Arrays cropped to the snippet window. Everything else is copied verbatim.
SPATIAL_ARRAYS = ("cwb", "era5", "XLAT", "XLONG")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="gs://norcorrdiff-us/taiwan_dataset/cwa_dataset/cwa_dataset.zarr",
        help="Source consolidated Zarr store (gs:// or local path).",
    )
    p.add_argument(
        "--dst",
        default="gs://norcorrdiff-us/taiwan_dataset/cwa_dataset/cwa_dataset_cut.zarr",
        help="Destination Zarr store to create (gs:// or local path).",
    )
    p.add_argument("--x-offset", type=int, default=250, help="snippet_x_offset")
    p.add_argument("--y-offset", type=int, default=50, help="snippet_y_offset")
    p.add_argument("--length", type=int, default=128, help="snippet_length")
    p.add_argument(
        "--workers", type=int, default=32, help="Concurrent timestep reads/writes."
    )
    p.add_argument(
        "--start", type=int, default=0, help="First timestep index (for resume)."
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="One past the last timestep index (default: all).",
    )
    p.add_argument(
        "--clevel", type=int, default=5, help="Blosc/lz4 compression level (0=off)."
    )
    return p.parse_args()


def open_store(path, mode):
    """Return a Zarr store/mapper for a gs:// or local path."""
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        return fs.get_mapper(path)
    return zarr.storage.DirectoryStore(path)


def main():
    args = parse_args()
    if int(zarr.__version__.split(".")[0]) >= 3:
        raise RuntimeError(
            f"zarr {zarr.__version__} detected, but this tool needs zarr v2: "
            "the source is a v2-format store, the v2 API is used here, and a "
            "v2 store stays readable by both zarr v2 and v3 (v3 output would "
            "not be readable by a v2 training container). "
            "Install with: pip install 'zarr<3'"
        )
    x0, y0, L = args.x_offset, args.y_offset, args.length
    ysl, xsl = slice(y0, y0 + L), slice(x0, x0 + L)

    resume = args.start > 0
    src = zarr.open_consolidated(open_store(args.src, "r"), mode="r")
    dst_store = open_store(args.dst, "w")
    dst = zarr.open_group(store=dst_store, mode="a")
    dst.attrs.update(dict(src.attrs))

    compressor = (
        Blosc(cname="lz4", clevel=args.clevel, shuffle=Blosc.SHUFFLE)
        if args.clevel > 0
        else None
    )

    # Validate the snippet fits inside the source grid.
    full_y, full_x = src["cwb"].shape[-2:]
    if y0 + L > full_y or x0 + L > full_x:
        raise ValueError(
            f"Snippet [{y0}:{y0 + L}, {x0}:{x0 + L}] exceeds source grid "
            f"({full_y}, {full_x})."
        )

    # On resume (--start > 0) keep already-written arrays; otherwise rebuild.
    if resume:
        logger.info("Resuming from timestep %d; keeping existing dst arrays", args.start)

    # 1) Copy every non-spatial array verbatim (preserves attrs + chunking).
    for key in src.array_keys():
        if key in SPATIAL_ARRAYS:
            continue
        if resume and key in dst:
            continue
        logger.info("Copying verbatim: %s", key)
        s = src[key]
        d = dst.create_dataset(
            key,
            shape=s.shape,
            chunks=s.chunks,
            dtype=s.dtype,
            compressor=s.compressor,
            fill_value=s.fill_value,
            overwrite=True,
        )
        d[...] = s[...]
        d.attrs.update(dict(s.attrs))

    # 2) Crop the small 2D coordinate grids in one shot.
    for key in ("XLAT", "XLONG"):
        if key not in src:
            continue
        if resume and key in dst:
            continue
        cropped = src[key][ysl, xsl]
        arr = dst.create_dataset(
            key,
            data=cropped,
            chunks=cropped.shape,
            compressor=compressor,
            overwrite=True,
        )
        arr.attrs.update(dict(src[key].attrs))

    # 3) Stream the big space-time arrays, one timestep per chunk.
    src_cwb, src_era5 = src["cwb"], src["era5"]
    T = src_cwb.shape[0]
    end = args.end if args.end is not None else T

    def make_array(name, src_arr):
        if resume and name in dst:
            return dst[name]
        arr = dst.create_dataset(
            name,
            shape=src_arr.shape[:2] + (L, L),
            chunks=(1,) + src_arr.shape[1:2] + (L, L),
            dtype=src_arr.dtype,
            fill_value=src_arr.fill_value,
            compressor=compressor,
            overwrite=True,
        )
        arr.attrs.update(dict(src_arr.attrs))
        return arr

    dst_cwb = make_array("cwb", src_cwb)
    dst_era5 = make_array("era5", src_era5)

    def process(i):
        dst_cwb[i] = src_cwb[i][:, ysl, xsl]
        dst_era5[i] = src_era5[i][:, ysl, xsl]

    logger.info(
        "Cropping %d timesteps [%d:%d] to %dx%d window at (y=%d, x=%d) "
        "with %d workers",
        end - args.start, args.start, end, L, L, y0, x0, args.workers,
    )
    t0 = time.time()
    done = 0
    block = max(args.workers * 2, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for bstart in range(args.start, end, block):
            bend = min(bstart + block, end)
            # .result() re-raises any worker exception.
            for _ in ex.map(process, range(bstart, bend)):
                pass
            done += bend - bstart
            rate = done / (time.time() - t0)
            logger.info(
                "%d/%d timesteps (%.1f/s, ETA %.0f min)",
                bstart - args.start + (bend - bstart),
                end - args.start,
                rate,
                (end - bend) / rate / 60 if rate > 0 else float("nan"),
            )

    # 4) Consolidate metadata so the store opens with zarr.open_consolidated.
    zarr.consolidate_metadata(dst_store)
    logger.info(
        "Done in %.1f min. Cut store: cwb%s era5%s",
        (time.time() - t0) / 60, dst_cwb.shape, dst_era5.shape,
    )


if __name__ == "__main__":
    main()
