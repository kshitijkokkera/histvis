"""Tests for histvis.utils"""
import numpy as np
import pytest
from pathlib import Path
from histvis.utils import load_npy, load_markers


def test_load_npy(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p = tmp_path / "test.npy"
    np.save(p, arr)
    result = load_npy(p)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, arr.astype(np.float32))


def test_load_markers_present(tmp_path):
    (tmp_path / "markers.txt").write_text("CD3\nCD8\nCD20\n")
    result = load_markers(tmp_path)
    assert result == ["CD3", "CD8", "CD20"]


def test_load_markers_missing(tmp_path):
    result = load_markers(tmp_path)
    assert result == []
