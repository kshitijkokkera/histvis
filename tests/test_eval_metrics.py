"""Tests for histvis.eval_metrics"""
import numpy as np
import pytest
from histvis.eval_metrics import calculate_metrics, _morans_i


def test_morans_i_uniform():
    arr = np.ones((4, 4))
    result = _morans_i(arr)
    assert np.isnan(result)  # zero variance → NaN


def test_morans_i_returns_float():
    rng = np.random.default_rng(42)
    arr = rng.random((8, 8)).astype(np.float32)
    result = _morans_i(arr)
    assert isinstance(result, float)


def test_calculate_metrics_basic(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.random((16, 16, 2)).astype(np.float32)
    pred = gt + rng.random((16, 16, 2)).astype(np.float32) * 0.1
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    np.save(run_dir / "gt_downsampled.npy", gt)
    np.save(run_dir / "prediction.npy", pred)
    (run_dir / "markers.txt").write_text("marker_A\nmarker_B\n")
    df = calculate_metrics(run_dirs=[str(run_dir)], output_dir=str(tmp_path / "metrics"))
    assert len(df) == 2
    assert "pearson_r" in df.columns
    assert "mae" in df.columns


def test_calculate_metrics_missing_files(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    df = calculate_metrics(run_dirs=[str(run_dir)], output_dir=str(tmp_path / "metrics"))
    assert df.empty
