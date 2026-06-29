import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


def plot_comparison(truth, pred, output_channels, time_label, channel_idx=0, cmap="viridis"):
    """Truth / prediction / error (pred - truth) for one sample/channel.

    Args:
        truth: (C, H, W) numpy array in physical units
        pred: (C, H, W) numpy array in physical units
        output_channels: list of channel objects with .name and .level
        time_label: string or datetime for the title
        channel_idx: which output channel to plot
        cmap: colormap for truth and prediction panels
    """
    t, p = truth[channel_idx], pred[channel_idx]
    err = p - t
    ch = output_channels[channel_idx]
    label = f"{ch.name} {ch.level}".strip()
    vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
    emax = np.abs(err).max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, arr, title, cm, kw in (
        (axes[0], t, "truth", cmap, dict(vmin=vmin, vmax=vmax)),
        (axes[1], p, "prediction", cmap, dict(vmin=vmin, vmax=vmax)),
        (axes[2], err, "error (pred - truth)", "RdBu_r", dict(vmin=-emax, vmax=emax)),
    ):
        im = ax.imshow(arr, cmap=cm, origin="upper", **kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    rmse = np.sqrt(np.mean(err**2))
    fig.suptitle(f"{label}  —  {time_label}   (RMSE={rmse:.3g})")
    plt.tight_layout()
    plt.show()


def plot_ensemble(truth, ensemble, output_channels, time_label, channel_idx=0, cmap="viridis"):
    """Truth / ensemble-mean / ensemble-std / mean-error for one diffusion sample/channel.

    Args:
        truth: (C, H, W) numpy array in physical units
        ensemble: (N, C, H, W) numpy array in physical units (N ensemble members)
        output_channels: list of channel objects with .name and .level
        time_label: string or datetime for the title
        channel_idx: which output channel to plot
        cmap: colormap for truth and mean panels
    """
    t = truth[channel_idx]
    ens = ensemble[:, channel_idx]  # (N, H, W)
    mean = ens.mean(axis=0)
    std = ens.std(axis=0)
    err = mean - t

    ch = output_channels[channel_idx]
    label = f"{ch.name} {ch.level}".strip()

    vmin, vmax = min(t.min(), mean.min()), max(t.max(), mean.max())
    emax = np.abs(err).max()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, arr, title, cm, kw in (
        (axes[0], t, "truth", cmap, dict(vmin=vmin, vmax=vmax)),
        (axes[1], mean, "ensemble mean", cmap, dict(vmin=vmin, vmax=vmax)),
        (axes[2], std, "ensemble std", "YlOrRd", {}),
        (axes[3], err, "mean error", "RdBu_r", dict(vmin=-emax, vmax=emax)),
    ):
        im = ax.imshow(arr, cmap=cm, origin="upper", **kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    rmse = np.sqrt(np.mean(err**2))
    fig.suptitle(f"{label}  —  {time_label}   (mean RMSE={rmse:.3g}, N={len(ensemble)})")
    plt.tight_layout()
    plt.show()


def plot_variable(dataset, source, channel_idx, sample_idx=0, predict_fn=None, cmap="RdBu_r"):
    """Plot a single channel from 'era5' (input), 'cwb' (target), or 'pred' (model output)
    in physical units.

    Args:
        dataset: DownscalingDataset
        source: "era5" | "cwb" | "pred"
        channel_idx: index into the selected channel subset (0-based)
        sample_idx: dataset sample index
        predict_fn: callable(idx) -> (truth, pred) in physical units; required for source="pred"
        cmap: colormap
    """
    if source == "pred":
        if predict_fn is None:
            raise ValueError("predict_fn is required for source='pred'")
        _, arr2d = predict_fn(sample_idx)
        arr = arr2d[channel_idx]
    elif source == "era5":
        arr = dataset.denormalize_input(dataset[sample_idx][1][None].numpy())[0, channel_idx]
    else:  # cwb
        arr = dataset.denormalize_output(dataset[sample_idx][0][None].numpy())[0, channel_idx]

    channels = dataset.input_channels() if source == "era5" else dataset.output_channels()
    ch = channels[channel_idx]
    label = f"{ch.name} {ch.level}".strip()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, label=f"{label} (physical units)")
    ax.set_title(f"{source}  [{label}]  —  sample {sample_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()


def animate_variable(
    dataset, source, channel_idx, predict_fn=None, cmap="RdBu_r", interval=100, start=0, n_steps=None
):
    """Animate a channel over a range of dataset samples in physical units.

    Args:
        dataset: DownscalingDataset
        source: "era5" | "cwb" | "pred"
        channel_idx: index into the selected channel subset (0-based).
            era5  -> dataset.input_channels()
            cwb / pred -> dataset.output_channels()
        predict_fn: callable(idx) -> (truth, pred) in physical units; required for source="pred".
            "pred" runs one forward pass per frame — fast for regression, slow for diffusion.
        interval: milliseconds between frames
        start: first dataset index
        n_steps: number of frames (None = all remaining)
    """
    channels = dataset.input_channels() if source == "era5" else dataset.output_channels()
    ch = channels[channel_idx]
    label = f"{ch.name} {ch.level}".strip()

    end = start + n_steps if n_steps is not None else len(dataset)
    times = dataset.time()[start:end]

    if source == "pred":
        if predict_fn is None:
            raise ValueError("predict_fn is required for source='pred'")
        all_frames = np.stack([predict_fn(i)[1][channel_idx] for i in range(start, end)])
    elif source == "era5":
        all_frames = np.stack([
            dataset.denormalize_input(dataset[i][1][None].numpy())[0, channel_idx]
            for i in range(start, end)
        ])
    else:  # cwb
        all_frames = np.stack([
            dataset.denormalize_output(dataset[i][0][None].numpy())[0, channel_idx]
            for i in range(start, end)
        ])
    vmin, vmax = all_frames.min(), all_frames.max()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(all_frames[0], cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"{label} (physical units)")
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()

    def update(i):
        im.set_data(all_frames[i])
        title.set_text(f"{source}  [{label}]  —  {times[i]}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=interval, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_wind_speed(dataset, source, predict_fn=None, cmap="viridis", interval=100, start=0, n_steps=None):
    """Animate 10m wind speed (sqrt(u² + v²)) in physical units.

    Args:
        dataset: DownscalingDataset
        source: "era5" | "cwb" | "pred"
        predict_fn: callable(idx) -> (truth, pred) in physical units; required for source="pred"
        interval: milliseconds between frames
        start: first dataset index
        n_steps: number of frames (None = all remaining)
    """
    channels = dataset.input_channels() if source == "era5" else dataset.output_channels()
    names = [ch.name for ch in channels]
    u_idx = names.index("eastward_wind_10m")
    v_idx = names.index("northward_wind_10m")

    end = start + n_steps if n_steps is not None else len(dataset)
    times = dataset.time()[start:end]

    if source == "pred":
        if predict_fn is None:
            raise ValueError("predict_fn is required for source='pred'")
        u_frames = np.stack([predict_fn(i)[1][u_idx] for i in range(start, end)])
        v_frames = np.stack([predict_fn(i)[1][v_idx] for i in range(start, end)])
    elif source == "era5":
        u_frames = np.stack([
            dataset.denormalize_input(dataset[i][1][None].numpy())[0, u_idx]
            for i in range(start, end)
        ])
        v_frames = np.stack([
            dataset.denormalize_input(dataset[i][1][None].numpy())[0, v_idx]
            for i in range(start, end)
        ])
    else:  # cwb
        u_frames = np.stack([
            dataset.denormalize_output(dataset[i][0][None].numpy())[0, u_idx]
            for i in range(start, end)
        ])
        v_frames = np.stack([
            dataset.denormalize_output(dataset[i][0][None].numpy())[0, v_idx]
            for i in range(start, end)
        ])

    speed_frames = np.sqrt(u_frames**2 + v_frames**2)
    vmin, vmax = speed_frames.min(), speed_frames.max()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(speed_frames[0], cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="wind speed (physical units)")
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()

    def update(i):
        im.set_data(speed_frames[i])
        title.set_text(f"{source}  [wind speed]  —  {times[i]}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(speed_frames), interval=interval, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())
