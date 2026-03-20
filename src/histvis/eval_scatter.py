"""
eval_scatter.py - Hexbin scatter plots for histvis.

Generates per-marker GT-vs-Predicted intensity hexbin scatter plots
and (optionally) side-by-side comparison grids when multiple prediction
arrays are provided.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _load_npy(path: Union[str, Path]) -> np.ndarray:
    arr = np.load(str(path))
    return arr.astype(np.float32)


def _load_markers(run_dir: Path) -> List[str]:
    markers_file = run_dir / "markers.txt"
    if markers_file.exists():
        return [m.strip() for m in markers_file.read_text().splitlines() if m.strip()]
    return []


def generate_scatter_plots(
    run_dirs: Union[str, List[str]],
    output_dir: Optional[str] = None,
    gt_filename: str = "gt_downsampled.npy",
    pred_filename: str = "prediction.npy",
    gridsize: int = 40,
    dpi: int = 150,
) -> Path:
    """Generate per-marker hexbin scatter plots (GT vs. Predicted).

    When multiple *run_dirs* are supplied, a side-by-side comparison figure
    is generated for each marker so you can visually compare models or ROIs.

    Parameters
    ----------
    run_dirs:
        A single run directory or list of run directories.  Each must contain
        ``gt_downsampled.npy`` and ``prediction.npy`` (or custom filenames).
    output_dir:
        Where to write the figures.  Defaults to ``<first_run_dir>/scatter``.
    gt_filename:
        Filename of the ground-truth array inside each run directory.
    pred_filename:
        Filename of the prediction array inside each run directory.
    gridsize:
        Number of hexagons in the x-direction for ``hexbin``.
    dpi:
        Resolution of saved figures.

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
    out_dir = Path(output_dir) if output_dir else run_paths[0] / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating scatter plots  n_dirs=%d", len(run_paths))

    # Collect valid data ---------------------------------------------------------
    gts: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    labels: List[str] = []

    for rp in run_paths:
        gt_path = rp / gt_filename
        pred_path = rp / pred_filename
        if not gt_path.exists() or not pred_path.exists():
            logger.warning("Skipping %s – required files not found", rp)
            continue
        gts.append(_load_npy(gt_path))
        preds.append(_load_npy(pred_path))
        labels.append(rp.name)

    if not gts:
        logger.error("No valid run directories found – aborting scatter plot generation")
        return out_dir

    markers = _load_markers(run_paths[0])
    n_markers = gts[0].shape[-1] if gts[0].ndim == 3 else 1
    if not markers:
        markers = [f"marker_{i}" for i in range(n_markers)]

    n_models = len(labels)

    for m_idx, marker_name in enumerate(markers):
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(5 * n_models, 5),
            squeeze=False,
        )
        fig.suptitle(f"{marker_name} – GT vs Predicted Intensity", fontsize=13)

        for col, (gt, pred, label) in enumerate(zip(gts, preds, labels)):
            gt_m = (gt[..., m_idx] if gt.ndim == 3 else gt).ravel()
            pred_m = (pred[..., m_idx] if pred.ndim == 3 else pred).ravel()

            ax = axes[0, col]
            hb = ax.hexbin(gt_m, pred_m, gridsize=gridsize, cmap="inferno", mincnt=1)
            fig.colorbar(hb, ax=ax, label="Count")

            # Identity line
            lims = [
                min(gt_m.min(), pred_m.min()),
                max(gt_m.max(), pred_m.max()),
            ]
            ax.plot(lims, lims, "w--", linewidth=1, alpha=0.7)

            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            ax.set_title(label)

        plt.tight_layout()
        safe_name = marker_name.replace("/", "_").replace(" ", "_")
        fig_path = out_dir / f"{safe_name}_scatter.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fig_path)

    logger.info("Scatter plot generation complete → %s", out_dir)
    return out_dir
