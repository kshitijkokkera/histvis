"""
inference.py - Core inference logic for histvis.

Wraps the helioscope-core validation/inference pipeline into a callable
function so the TUI can invoke it directly without subprocess.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def run_inference(
    slide_id: str,
    region_coords: str,
    checkpoint_path: str,
    output_dir: str,
    config_file: Optional[str] = None,
    summary_path: Optional[str] = None,
    split_file: Optional[str] = None,
    device: str = "cuda",
) -> Path:
    """Run helioscope-core inference for a single (slide, ROI, model) pair.

    Parameters
    ----------
    slide_id:
        Identifier of the whole-slide image to process.
    region_coords:
        Space-separated corner coordinates of the region-of-interest,
        e.g. ``"19580,15537 19580,23537 11580,23537 11580,15537"``.
    checkpoint_path:
        Path to the PyTorch ``.pth`` checkpoint file.
    output_dir:
        Directory where ``prediction.npy`` and ``gt_downsampled.npy``
        will be written.
    config_file:
        Optional path to the model config / GVHD text file.
    summary_path:
        Optional path to the dataset summary CSV.
    split_file:
        Optional path to the train/val/test split file.
    device:
        Torch device string (``"cuda"`` or ``"cpu"``).

    Returns
    -------
    Path
        The output directory that was written to.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Starting inference for slide %s", slide_id)
    logger.info("  Checkpoint : %s", checkpoint_path)
    logger.info("  ROI coords : %s", region_coords)
    logger.info("  Output dir : %s", out)

    try:
        import torch
        import numpy as np

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        logger.info("Loading model from checkpoint …")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # TODO: Instantiate and load the model once helioscope-core is integrated.
        # state_dict = (
        #     checkpoint.get("model_state_dict")
        #     or checkpoint.get("state_dict")
        #     or checkpoint
        # )

        # ------------------------------------------------------------------
        # Parse region coordinates
        # ------------------------------------------------------------------
        coords: List[Tuple[int, int]] = []
        for pair in region_coords.strip().split():
            x_str, y_str = pair.split(",")
            coords.append((int(x_str), int(y_str)))
        logger.info("Parsed %d coordinate pairs", len(coords))

        # ------------------------------------------------------------------
        # Placeholder: replace the block below with actual WSI loading and
        # model forward-pass once helioscope-core is integrated.
        # ------------------------------------------------------------------
        logger.warning(
            "helioscope-core integration stub reached – writing placeholder arrays."
        )
        _PLACEHOLDER_IMAGE_SIZE = 256
        _PLACEHOLDER_MARKER_COUNT = 10
        prediction = np.zeros(
            (_PLACEHOLDER_IMAGE_SIZE, _PLACEHOLDER_IMAGE_SIZE, _PLACEHOLDER_MARKER_COUNT),
            dtype=np.float32,
        )
        gt_downsampled = np.zeros(
            (_PLACEHOLDER_IMAGE_SIZE, _PLACEHOLDER_IMAGE_SIZE, _PLACEHOLDER_MARKER_COUNT),
            dtype=np.float32,
        )

        np.save(out / "prediction.npy", prediction)
        np.save(out / "gt_downsampled.npy", gt_downsampled)

        markers_path = out / "markers.txt"
        if not markers_path.exists():
            markers_path.write_text(
                "\n".join(f"marker_{i}" for i in range(_PLACEHOLDER_MARKER_COUNT))
            )

        logger.info("Inference complete.  Outputs written to %s", out)

    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise

    return out


def run_inference_batch(
    experiments: List[dict],
    output_base_dir: str,
    device: str = "cuda",
) -> List[Path]:
    """Run inference for a list of (slide, model) experiment configurations.

    Each element of *experiments* is a ``dict`` with the keys accepted by
    :func:`run_inference`.  Results are placed in automatically named
    sub-directories under *output_base_dir*.

    Parameters
    ----------
    experiments:
        List of experiment parameter dicts.
    output_base_dir:
        Root directory under which per-experiment sub-directories are created.
    device:
        Torch device string.

    Returns
    -------
    List[Path]
        One output path per experiment in the same order as *experiments*.
    """
    base = Path(output_base_dir)
    results: List[Path] = []

    for idx, exp in enumerate(experiments):
        slide_id = exp.get("slide_id", f"slide_{idx}")
        model_name = Path(exp.get("checkpoint_path", f"model_{idx}")).stem
        sub_dir = base / f"slide_{slide_id}" / f"pred_{model_name}"

        out_path = run_inference(
            slide_id=slide_id,
            region_coords=exp.get("region_coords", ""),
            checkpoint_path=exp.get("checkpoint_path", ""),
            output_dir=str(sub_dir),
            config_file=exp.get("config_file"),
            summary_path=exp.get("summary_path"),
            split_file=exp.get("split_file"),
            device=device,
        )
        results.append(out_path)

    return results
