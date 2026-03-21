"""
eval_errors.py - Error heatmaps and histograms for histvis.

Generates per-marker absolute-error and squared-error visualisations
comparing one or more predictions against one or more ground-truth arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from histvis.utils import load_npy, load_markers

logger = logging.getLogger(__name__)


def generate_error_maps(
    run_dirs: Union[str, List[str]],
    output_dir: Optional[str] = None,
    metric: str = "mae",
    gt_filename: str = "gt_downsampled.npy",
    pred_filename: str = "prediction.npy",
    dpi: int = 150,
) -> Path:
    """Generate error heatmaps and histograms for one or more run directories.

    Parameters
    ----------
    run_dirs:
        A single run directory (string or Path) or a list of run directories.
        Each directory must contain ``gt_downsampled.npy`` and
        ``prediction.npy`` (or the filenames given by *gt_filename* /
        *pred_filename*).
    output_dir:
        Where to write the output figures.  Defaults to the first run_dir.
    metric:
        Error metric to plot: ``"mae"`` (mean absolute error) or
        ``"rmse"`` (root-mean-squared error).
    gt_filename:
        Filename of the ground-truth array inside each run directory.
    pred_filename:
        Filename of the prediction array inside each run directory.
    dpi:
        Resolution of the saved figures.

    Returns
    -------
    Path
        Directory containing the generated figures.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]

    run_paths = [Path(d) for d in run_dirs]
    out_dir = Path(output_dir) if output_dir else run_paths[0] / "error_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating error maps  metric=%s  n_dirs=%d", metric, len(run_paths))

    # Gather data ----------------------------------------------------------------
    gts: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    labels: List[str] = []

    for rp in run_paths:
        gt_path = rp / gt_filename
        pred_path = rp / pred_filename
        if not gt_path.exists() or not pred_path.exists():
            logger.warning("Skipping %s – required files not found", rp)
            continue
        gts.append(load_npy(gt_path))
        preds.append(load_npy(pred_path))
        labels.append(rp.name)

    if not gts:
        logger.error("No valid run directories found – aborting error map generation")
        return out_dir

    # Use marker names from the first run dir ------------------------------------
    markers = load_markers(run_paths[0])
    n_markers = gts[0].shape[-1] if gts[0].ndim == 3 else 1

    if not markers:
        markers = [f"marker_{i}" for i in range(n_markers)]

    # Plot per marker ------------------------------------------------------------
    n_preds = len(preds)
    for m_idx, marker_name in enumerate(markers):
        fig, axes = plt.subplots(
            2, n_preds + 1,
            figsize=(4 * (n_preds + 1), 8),
            squeeze=False,
        )
        fig.suptitle(f"{marker_name} – {metric.upper()} Error Maps", fontsize=14)

        gt_slice = gts[0][..., m_idx] if gts[0].ndim == 3 else gts[0]
        axes[0, 0].imshow(gt_slice, cmap="viridis")
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].axis("off")
        axes[1, 0].hist(gt_slice.ravel(), bins=50, color="steelblue", edgecolor="none")
        axes[1, 0].set_title("GT Distribution")
        axes[1, 0].set_xlabel("Intensity")

        vmax_error = 0.0
        error_slices = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            pred_slice = pred[..., m_idx] if pred.ndim == 3 else pred
            if metric == "rmse":
                err = np.sqrt((pred_slice - gt_slice) ** 2)
            else:
                err = np.abs(pred_slice - gt_slice)
            error_slices.append((err, label))
            vmax_error = max(vmax_error, float(err.max()))

        norm = mcolors.Normalize(vmin=0, vmax=vmax_error if vmax_error > 0 else 1)
        for col, (err, label) in enumerate(error_slices, start=1):
            im = axes[0, col].imshow(err, cmap="hot", norm=norm)
            axes[0, col].set_title(f"{label}")
            axes[0, col].axis("off")
            fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
            axes[1, col].hist(err.ravel(), bins=50, color="tomato", edgecolor="none")
            axes[1, col].set_title(f"{label} Error Dist.")
            axes[1, col].set_xlabel(metric.upper())

        plt.tight_layout()
        safe_name = marker_name.replace("/", "_").replace(" ", "_")
        fig_path = out_dir / f"{safe_name}_{metric}_error.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fig_path)

    logger.info("Error map generation complete → %s", out_dir)
    return out_dir
