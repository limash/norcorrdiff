"""
Regression-only inference entry point.

Loads a regression network from a checkpoint and runs inference for every
time step in the dataset that falls within [generation.start_time,
generation.end_time] (ISO 8601, inclusive). Results are written to a NetCDF
file.

Example
-------
python generate_regression.py \\
    generation.start_time=2012-06-02T00:00:00 \\
    generation.end_time=2012-06-04T18:00:00 \\
    generation.io.reg_ckpt_filename=/path/to/CorrDiffRegressionUNet.mdlus \\
    generation.io.output_filename=results/regression_output.nc
"""

import datetime
import os
import tempfile

import cftime
import fsspec
import hydra
import netCDF4 as nc
import torch
from datasets.dataset import init_dataset_from_config, register_dataset
from helpers.generate_helpers import NetCDFWriter, save_images
from helpers.train_helpers import _convert_datetime_to_cftime
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from physicsnemo import Module
from physicsnemo.diffusion.generate import regression_step
from physicsnemo.distributed import DistributedManager


def _load_regression_net(path: str):
    """Load a Module from a checkpoint, downloading from GCS to a temp file if needed."""
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile(suffix=".mdlus", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with fsspec.open(path, "rb") as remote_f:
                with open(tmp_path, "wb") as local_f:
                    local_f.write(remote_f.read())
            return Module.from_checkpoint(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        return Module.from_checkpoint(to_absolute_path(path))


def _parse_time(s: str) -> cftime.DatetimeGregorian:
    return _convert_datetime_to_cftime(
        datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    )


def _setup_input_wind_speed(writer, component_indices):
    x_u10_idx, x_v10_idx = component_indices
    writer.input_group.createVariable("wind_speed", "f", dimensions=("time", "y", "x"))
    return (x_u10_idx, x_v10_idx)


def _write_input_wind_speed(writer, dataset, image_lr, time_index, component_indices):
    x_u10_idx, x_v10_idx = component_indices
    image_lr_np = dataset.denormalize_input(image_lr.cpu().numpy())
    input_u10 = image_lr_np[0, x_u10_idx]
    input_v10 = image_lr_np[0, x_v10_idx]
    input_wind_speed = (input_u10**2 + input_v10**2) ** 0.5
    writer.write_input("wind_speed", time_index, input_wind_speed)


def _setup_wind_speed_output(writer, output_channels):
    y_u10_idx, y_v10_idx = output_channels
    writer.truth_group.createVariable("wind_speed", "f", dimensions=("time", "y", "x"))
    writer.prediction_group.createVariable(
        "wind_speed", "f", dimensions=("ensemble", "time", "y", "x")
    )
    return (y_u10_idx, y_v10_idx)


def _write_wind_speed(writer, dataset, image_tar, image_out, time_index, component_indices):
    y_u10_idx, y_v10_idx = component_indices
    image_tar_np = dataset.denormalize_output(image_tar.cpu().numpy())
    truth_u10 = image_tar_np[0, y_u10_idx]
    truth_v10 = image_tar_np[0, y_v10_idx]
    truth_wind_speed = (truth_u10**2 + truth_v10**2) ** 0.5
    writer.write_truth("wind_speed", time_index, truth_wind_speed)

    image_out_np = dataset.denormalize_output(image_out.cpu().numpy())
    for ensemble_index in range(image_out_np.shape[0]):
        pred_u10 = image_out_np[ensemble_index, y_u10_idx]
        pred_v10 = image_out_np[ensemble_index, y_v10_idx]
        pred_wind_speed = (pred_u10**2 + pred_v10**2) ** 0.5
        writer.write_prediction(
            "wind_speed", time_index, ensemble_index, pred_wind_speed
        )


@hydra.main(
    version_base="1.2",
    config_path="conf",
    config_name="config_generate_regression_carra2.yaml",
)
def main(cfg: DictConfig) -> None:
    print("Initializing distributed manager...")
    DistributedManager.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset ------------------------------------------------------------
    print("Loading dataset...")
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    print(f"Dataset loaded. Available time steps: {len(dataset.time())}")

    # Filter dataset times to the requested range
    print(f"Filtering times between {cfg.generation.start_time} and {cfg.generation.end_time}...")
    start = _parse_time(cfg.generation.start_time)
    end = _parse_time(cfg.generation.end_time)
    all_times = dataset.time()
    sampler = [i for i, t in enumerate(all_times) if start <= t <= end]
    if not sampler:
        raise ValueError(
            f"No time steps found in dataset between {cfg.generation.start_time} "
            f"and {cfg.generation.end_time}."
        )
    print(f"Selected {len(sampler)} time steps for inference")

    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    print(f"Image shape: {img_shape}, output channels: {img_out_channels}")

    # ---- Regression network -------------------------------------------------
    print(f"Loading regression network from {cfg.generation.io.reg_ckpt_filename}...")
    net_reg = _load_regression_net(cfg.generation.io.reg_ckpt_filename)
    net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
    if hasattr(net_reg, "amp_mode"):
        net_reg.amp_mode = False
    print("Regression network loaded")

    # ---- Output file --------------------------------------------------------
    print(f"Creating output file: {cfg.generation.io.output_filename}")
    f = nc.Dataset(cfg.generation.io.output_filename, "w")
    f.cfg = str(cfg)
    writer = NetCDFWriter(
        f,
        lat=dataset.latitude(),
        lon=dataset.longitude(),
        input_channels=dataset.input_channels(),
        output_channels=dataset.output_channels(),
    )
    input_channel_names = [
        channel.name + channel.level for channel in dataset.input_channels()
    ]
    input_wind_component_indices = _setup_input_wind_speed(
        writer,
        (
            input_channel_names.index("x_u10"),
            input_channel_names.index("x_v10"),
        ),
    )

    output_channel_names = [
        channel.name + channel.level for channel in dataset.output_channels()
    ]
    wind_component_indices = _setup_wind_speed_output(
        writer,
        (
            output_channel_names.index("y_u10"),
            output_channel_names.index("y_v10"),
        ),
    )

    # ---- Inference loop -----------------------------------------------------
    print("Starting inference loop...")
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
    )

    with torch.no_grad():
        for time_index, (dataset_index, (image_tar, image_lr)) in enumerate(
            zip(sampler, data_loader)
        ):
            time_value = all_times[dataset_index]
            print(
                f"  [{time_index + 1}/{len(sampler)}] Processing time step "
                f"{dataset_index} at {time_value}..."
            )
            image_lr = (
                image_lr.to(device=device)
                .to(torch.float32)
                .to(memory_format=torch.channels_last)
            )
            image_tar = image_tar.to(device=device).to(torch.float32)

            image_out = regression_step(
                net=net_reg,
                img_lr=image_lr,
                latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                lead_time_label=None,
            )

            save_images(
                writer,
                dataset,
                all_times,
                image_out.cpu(),
                image_tar.cpu(),
                image_lr.cpu(),
                time_index,
                dataset_index,
            )

            _write_input_wind_speed(
                writer,
                dataset,
                image_lr,
                time_index,
                input_wind_component_indices,
            )
            _write_wind_speed(
                writer,
                dataset,
                image_tar,
                image_out,
                time_index,
                wind_component_indices,
            )

    f.close()
    print("Inference complete. Output saved.")


if __name__ == "__main__":
    main()
