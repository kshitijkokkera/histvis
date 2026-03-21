# histvis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Helioscope TUI** — Terminal UI for helioscope-core inference and analysis.

`histvis` is a keyboard-driven terminal application that lets you run digital
pathology inference and visualisation workflows without leaving the command
line. It wraps the `helioscope-core` validation pipeline and provides live
log output, statistical metrics, scatter plots, and error maps — all from a
single `histvis` command.

---

## Features

- **3 workflow modes** — Full (Inference → Metrics → Graphs), Inference Only, Analysis Only
- **4 comparison scenarios** — Single Run, Model Comparison, Batch Processing, Full Matrix
- **Live log panel** — real-time progress streamed to a scrollable Rich log widget
- **Statistical metrics** — Pearson r, Spearman r, R², MAE, RMSE, Wasserstein distance, Moran's I
- **Hexbin scatter plots** — GT vs. Predicted intensity per marker
- **Error heatmaps** — per-marker MAE/RMSE error maps with distribution histograms
- **Dynamic entry lists** — add/remove slide and model entries on the fly

---

## Installation

Requires **Python ≥ 3.9**.

```bash
pip install -e .
```

---

## Usage

```bash
histvis
```

The TUI opens in your terminal. Use the tabs to configure your workflow, then
press **▶ Run Selected Workflow** to start.

| Key      | Action            |
|----------|-------------------|
| `q`      | Quit              |
| `d`      | Toggle dark mode  |
| `Ctrl+L` | Clear log         |

---

## Workflow Modes

| Mode              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| **Full Workflow** | Runs inference, then computes metrics and generates graphs.  |
| **Inference Only**| Runs the PyTorch inference step and writes `.npy` outputs.   |
| **Analysis Only** | Skips inference; reads existing `.npy` files and plots them. |

---

## Comparison Scenarios

| Scenario              | Slides | Models | Description                                    |
|-----------------------|--------|--------|------------------------------------------------|
| **Single Run**        | 1      | 1      | Baseline single-slide, single-model run.       |
| **Model Comparison**  | 1      | N      | Compare multiple models on the same slide.     |
| **Batch Processing**  | N      | 1      | Run one model across many slides or ROIs.      |
| **Full Matrix**       | N      | N      | Every slide paired with every model.           |

---

## Dependencies

- [textual](https://github.com/Textualize/textual) ≥ 0.47.0
- [numpy](https://numpy.org/) ≥ 1.24.0
- [matplotlib](https://matplotlib.org/) ≥ 3.7.0
- [scipy](https://scipy.org/) ≥ 1.10.0
- [pandas](https://pandas.pydata.org/) ≥ 2.0.0
- [torch](https://pytorch.org/) ≥ 2.0.0
- [torchvision](https://pytorch.org/vision/) ≥ 0.15.0
- [tifffile](https://github.com/cgohlke/tifffile) ≥ 2023.1.1
- [zarr](https://zarr.readthedocs.io/) ≥ 2.14.0
- [opencv-python](https://opencv.org/) ≥ 4.7.0
- [tqdm](https://tqdm.github.io/) ≥ 4.65.0
- [imagecodecs](https://github.com/cgohlke/imagecodecs) ≥ 2023.1.1
- [scikit-learn](https://scikit-learn.org/) ≥ 1.2.0

---

## License

[MIT](LICENSE) © 2026 kshitijkokkera
