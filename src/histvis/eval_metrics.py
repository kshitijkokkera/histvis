"""
eval_metrics.py - Statistical metrics computation for histvis.

Computes per-marker Pearson correlation, Spearman correlation, R²,
Wasserstein distance, and Moran's I spatial autocorrelation between
one or more prediction arrays and one or more ground-truth arrays.
Results are written to a CSV and returned as a DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from histvis.utils import load_npy, load_markers

logger = logging.getLogger(__name__)


def _morans_i(arr: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Moran's I spatial autocorrelation for a 2-D array."""
    from scipy.spatial.distance import cdist

    flat = arr.ravel()
    n = len(flat)
    if n < 4:
        return float("nan")

    h, w = arr.shape
    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.column_stack([ys.ravel(), xs.ravel()])

    inv_dist = 1.0 / (cdist(coords, coords, metric="euclidean") + eps)
    np.fill_diagonal(inv_dist, 0.0)
    W = inv_dist / inv_dist.sum()

    z = flat - flat.mean()
    numerator = n * float(z @ W @ z)
    denominator = float(z @ z)
    return numerator / denominator if denominator != 0 else float("nan")


def _metrics_for_pair(
    gt: np.ndarray,
    pred: np.ndarray,
    marker: str,
    label: str,
) -> Dict[str, object]:
    """Compute metrics for a single (gt, pred, marker) triplet."""
    from scipy.stats import pearsonr, spearmanr, wasserstein_distance
    from sklearn.metrics import r2_score  # type: ignore[import]

    gt_flat = gt.ravel()
    pred_flat = pred.ravel()

    pearson_r, _ = pearsonr(gt_flat, pred_flat)
    spearman_r, _ = spearmanr(gt_flat, pred_flat)
    r2 = r2_score(gt_flat, pred_flat)
    wass = wasserstein_distance(gt_flat, pred_flat)
    mae = float(np.abs(gt_flat - pred_flat).mean())
    rmse = float(np.sqrt(((gt_flat - pred_flat) ** 2).mean()))
    morans = _morans_i(pred.reshape(gt.shape) if pred.ndim == 1 else pred)

    return {
        "marker": marker,
        "model": label,
        "pearson_r": round(float(pearson_r), 6),
        "spearman_r": round(float(spearman_r), 6),
        "r2": round(float(r2), 6),
        "wasserstein": round(float(wass), 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "morans_i": round(morans, 6) if not np.isnan(morans) else None,
    }


def calculate_metrics(
    run_dirs: Union[str, List[str]],
    output_dir: Optional[str] = None,
    gt_filename: str = "gt_downsampled.npy",
    pred_filename: str = "prediction.npy",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Compute comprehensive statistical metrics for one or more run directories.

    Parameters
    ----------
    run_dirs:
        A single run directory or list of run directories.  Each must contain
        ``gt_downsampled.npy`` and ``prediction.npy`` (or the custom filenames).
    output_dir:
        Where to write ``metrics_per_marker.csv``.  Defaults to first run_dir.
    gt_filename:
        Filename of the ground-truth array inside each run directory.
    pred_filename:
        Filename of the prediction array inside each run directory.
    eps:
        Small constant used in Moran's I distance computation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: marker, model, pearson_r, spearman_r, r2,
        wasserstein, mae, rmse, morans_i.
    """
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]

    run_paths = [Path(d) for d in run_dirs]
    out_dir = Path(output_dir) if output_dir else run_paths[0] / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Calculating metrics  n_dirs=%d", len(run_paths))

    rows: List[Dict] = []

    for rp in run_paths:
        gt_path = rp / gt_filename
        pred_path = rp / pred_filename
        if not gt_path.exists() or not pred_path.exists():
            logger.warning("Skipping %s – required files not found", rp)
            continue

        gt = load_npy(gt_path)
        pred = load_npy(pred_path)
        markers = load_markers(rp)
        n_markers = gt.shape[-1] if gt.ndim == 3 else 1
        if not markers:
            markers = [f"marker_{i}" for i in range(n_markers)]

        for m_idx, marker in enumerate(markers):
            gt_m = gt[..., m_idx] if gt.ndim == 3 else gt
            pred_m = pred[..., m_idx] if pred.ndim == 3 else pred
            try:
                row = _metrics_for_pair(gt_m, pred_m, marker, rp.name)
                rows.append(row)
                logger.info("  %s / %s  R²=%.4f  MAE=%.4f", rp.name, marker, row["r2"], row["mae"])
            except Exception as exc:
                logger.error("  Failed for %s / %s: %s", rp.name, marker, exc)

    if not rows:
        logger.warning("No metrics computed – returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_per_marker.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Metrics written to %s", csv_path)
    return df
