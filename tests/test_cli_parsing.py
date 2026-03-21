"""Tests for CLI entry parsing helpers (non-TUI logic)."""
import pytest


def test_slide_entry_parsing():
    """Validate the pipe-separated parsing logic used in _collect_slide_entries."""
    raw = "slide_42 | 100,200 300,400 500,600 700,800 | /data/gt.npy"
    parts = [p.strip() for p in raw.split("|")]
    assert parts[0] == "slide_42"
    assert parts[1] == "100,200 300,400 500,600 700,800"
    assert parts[2] == "/data/gt.npy"


def test_model_entry_parsing():
    """Validate the pipe-separated parsing logic used in _collect_model_entries."""
    raw = "/checkpoints/model.pth | MyModel"
    parts = [p.strip() for p in raw.split("|")]
    assert parts[0] == "/checkpoints/model.pth"
    assert parts[1] == "MyModel"


def test_model_entry_no_label():
    """Model entry with no label falls back to stem of checkpoint path."""
    from pathlib import Path
    raw = "/checkpoints/convnext_epoch9.pth"
    parts = [p.strip() for p in raw.split("|")]
    checkpoint_path = parts[0]
    label = parts[1] if len(parts) > 1 else Path(checkpoint_path).stem
    assert label == "convnext_epoch9"
