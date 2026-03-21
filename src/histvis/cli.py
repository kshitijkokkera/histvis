"""
cli.py - Textual TUI for histvis.

Entry point: ``histvis`` (defined in pyproject.toml).

The application supports three workflow modes:
  1. Full Workflow  – Inference ➜ Metrics ➜ Graphs
  2. Inference Only – Only run the PyTorch inference step
  3. Analysis Only  – Skip inference; run metrics/graphs on existing .npy files

Scenario support (multiple GTs / multiple Preds):
  A. Model Comparison  – 1 GT, N Models
  B. Batch Processing  – N Data (slides/ROIs), 1 Model
  C. Full Matrix       – N Data, N Models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Rule,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging bridge – routes Python logging records into a RichLog widget
# ---------------------------------------------------------------------------

class _RichLogHandler(logging.Handler):
    """Forwards log records to a Textual RichLog widget."""

    def __init__(self, log_widget: RichLog) -> None:
        super().__init__()
        self._widget = log_widget

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        msg = self.format(record)
        colour_map = {
            "DEBUG": "dim white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        colour = colour_map.get(level, "white")
        self._widget.write(f"[{colour}]{msg}[/{colour}]")


# ---------------------------------------------------------------------------
# Helper: collectable multi-entry list (simple list of Input rows)
# ---------------------------------------------------------------------------

class _EntryRow(Horizontal):
    """A single row in a dynamic list: one text input + a remove button."""

    DEFAULT_CSS = """
    _EntryRow {
        height: auto;
        margin-bottom: 1;
    }
    _EntryRow Input {
        width: 1fr;
    }
    _EntryRow Button {
        width: 5;
        min-width: 5;
        margin-left: 1;
    }
    """

    def __init__(self, placeholder: str = "", initial_value: str = "", row_id: str = "") -> None:
        super().__init__()
        self._placeholder = placeholder
        self._initial_value = initial_value
        self._row_id = row_id

    def compose(self) -> ComposeResult:
        yield Input(value=self._initial_value, placeholder=self._placeholder, id=f"entry_{self._row_id}")
        yield Button("✕", variant="error", id=f"remove_{self._row_id}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class HistVisApp(App):
    """Textual TUI for helioscope-core inference and analysis."""

    TITLE = "histvis – Helioscope Validation & Analysis"
    SUB_TITLE = "Digital Pathology Visualization Tool"

    CSS = """
    Screen {
        layout: vertical;
    }

    /* ---- Tab content ---- */
    .tab-content {
        padding: 1 2;
        height: 1fr;
    }

    /* ---- Section labels ---- */
    .section-label {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    /* ---- Input rows ---- */
    .field-row {
        height: auto;
        margin-bottom: 1;
    }
    .field-label {
        width: 24;
        content-align: right middle;
        padding-right: 1;
        color: $text-muted;
    }
    .field-input {
        width: 1fr;
    }

    /* ---- Workflow selector ---- */
    #workflow-radio {
        height: auto;
        margin-bottom: 1;
    }

    /* ---- Scenario selector ---- */
    #scenario-select {
        width: 1fr;
    }

    /* ---- Dynamic lists ---- */
    .entry-list {
        height: auto;
        border: solid $surface-lighten-2;
        padding: 1;
        margin-bottom: 1;
    }
    .list-header {
        text-style: bold underline;
        margin-bottom: 1;
    }
    .add-btn {
        width: auto;
        margin-top: 1;
    }

    /* ---- Run button ---- */
    #run-btn {
        width: 100%;
        height: 4;
        margin-top: 1;
        text-style: bold;
    }

    /* ---- Log panel ---- */
    #log-panel {
        height: 14;
        border: solid $primary;
        margin-top: 1;
    }

    /* ---- Disabled styling ---- */
    .hidden {
        display: none;
    }

    /* ---- Info text ---- */
    .info-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
        Binding("ctrl+l", "clear_log", "Clear Log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._slide_row_count = 0
        self._model_row_count = 0
        self._inference_rows: List[str] = []  # row ids for slide entries
        self._model_rows: List[str] = []       # row ids for model entries
        self._log_handler: Optional[_RichLogHandler] = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent(id="tabs"):

            # ---- Tab 1: Workflow ----------------------------------------
            with TabPane("⚙ Workflow", id="tab-workflow"):
                with ScrollableContainer(classes="tab-content"):
                    yield Label("Workflow Mode", classes="section-label")
                    with RadioSet(id="workflow-radio"):
                        yield RadioButton("Full Workflow  (Inference ➜ Metrics ➜ Graphs)", value=True, id="mode-full")
                        yield RadioButton("Inference Only", id="mode-infer")
                        yield RadioButton("Analysis Only  (Use existing .npy files)", id="mode-analysis")

                    yield Rule()

                    # ---- Scenario ----------------------------------------
                    yield Label("Comparison Scenario", classes="section-label")
                    yield Select(
                        options=[
                            ("Single Run  (1 slide, 1 model)", "single"),
                            ("Model Comparison  (1 slide, N models)", "model_compare"),
                            ("Batch Processing  (N slides, 1 model)", "batch"),
                            ("Full Matrix  (N slides, N models)", "matrix"),
                        ],
                        value="single",
                        id="scenario-select",
                    )

                    yield Rule()

                    # ---- Common inputs -----------------------------------
                    yield Label("Common Configuration", classes="section-label")
                    with Horizontal(classes="field-row"):
                        yield Label("Output Directory", classes="field-label")
                        yield Input(placeholder="/path/to/output", id="output-dir", classes="field-input")

                    # ---- Inference-specific inputs -----------------------
                    with Container(id="inference-section"):
                        yield Label("Inference Inputs", classes="section-label")
                        yield Static(
                            "These fields are used when running inference.",
                            classes="info-text",
                        )

                        with Horizontal(classes="field-row"):
                            yield Label("Config File", classes="field-label")
                            yield Input(placeholder="/path/to/config.txt", id="config-file", classes="field-input")

                        with Horizontal(classes="field-row"):
                            yield Label("Summary Path", classes="field-label")
                            yield Input(placeholder="/path/to/summary.csv", id="summary-path", classes="field-input")

                        with Horizontal(classes="field-row"):
                            yield Label("Split File", classes="field-label")
                            yield Input(placeholder="/path/to/split.txt", id="split-file", classes="field-input")

                        with Horizontal(classes="field-row"):
                            yield Label("Device", classes="field-label")
                            yield Input(value="cuda", id="device", classes="field-input")

                    yield Rule()
                    yield Button("▶  Run Selected Workflow", variant="success", id="run-btn")

            # ---- Tab 2: Data Sources (Slides / GT) ----------------------
            with TabPane("📂 Data Sources", id="tab-data"):
                with ScrollableContainer(classes="tab-content"):
                    yield Label("Slide / Ground-Truth Entries", classes="section-label")
                    yield Static(
                        "Add one entry per slide or ROI.  Each entry encodes\n"
                        "  slide_id | region_coords | gt_npy_path (optional)\n"
                        "separated by  |  characters.",
                        classes="info-text",
                    )
                    with Vertical(id="slide-list", classes="entry-list"):
                        yield Label("Slide Entries", classes="list-header")
                    yield Button("+ Add Slide Entry", id="add-slide", classes="add-btn")

            # ---- Tab 3: Models (Checkpoints / Preds) --------------------
            with TabPane("🤖 Models", id="tab-models"):
                with ScrollableContainer(classes="tab-content"):
                    yield Label("Model / Checkpoint Entries", classes="section-label")
                    yield Static(
                        "Add one entry per model checkpoint.  Each entry encodes\n"
                        "  checkpoint_path | label (optional)\n"
                        "separated by  |  characters.  For Analysis Only mode you\n"
                        "can provide a path to an existing  prediction.npy  file.",
                        classes="info-text",
                    )
                    with Vertical(id="model-list", classes="entry-list"):
                        yield Label("Model Entries", classes="list-header")
                    yield Button("+ Add Model Entry", id="add-model", classes="add-btn")

            # ---- Tab 4: Live Log ----------------------------------------
            with TabPane("📋 Log", id="tab-log"):
                with Container(classes="tab-content"):
                    yield Label("Live Output Log", classes="section-label")
                    yield RichLog(id="log-output", highlight=True, markup=True)
                    with Horizontal():
                        yield Button("Clear Log", id="clear-log-btn", variant="default")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Wire up logging bridge and populate default rows."""
        log_widget = self.query_one("#log-output", RichLog)
        self._log_handler = _RichLogHandler(log_widget)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self._log_handler)
        root_logger.setLevel(logging.DEBUG)

        # Add one default slide and model row
        self._add_slide_row()
        self._add_model_row()

    # ------------------------------------------------------------------
    # Dynamic list helpers
    # ------------------------------------------------------------------

    def _add_slide_row(self, value: str = "") -> None:
        self._slide_row_count += 1
        row_id = f"slide_{self._slide_row_count}"
        self._inference_rows.append(row_id)
        list_widget = self.query_one("#slide-list", Vertical)
        row = _EntryRow(
            placeholder="slide_2 | x1,y1 x2,y2 x3,y3 x4,y4 | /optional/path/to/gt.npy",
            initial_value=value,
            row_id=row_id,
        )
        list_widget.mount(row)

    def _add_model_row(self, value: str = "") -> None:
        self._model_row_count += 1
        row_id = f"model_{self._model_row_count}"
        self._model_rows.append(row_id)
        list_widget = self.query_one("#model-list", Vertical)
        row = _EntryRow(
            placeholder="/path/to/checkpoint.pth | ConvNeXt-epoch9",
            initial_value=value,
            row_id=row_id,
        )
        list_widget.mount(row)

    def _remove_entry_row(self, row_id: str) -> None:
        try:
            row_widget = self.query_one(f"#entry_{row_id}").parent
            row_widget.remove()
        except Exception:
            pass
        if row_id in self._inference_rows:
            self._inference_rows.remove(row_id)
        if row_id in self._model_rows:
            self._model_rows.remove(row_id)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "add-slide":
            self._add_slide_row()
        elif btn_id == "add-model":
            self._add_model_row()
        elif btn_id and btn_id.startswith("remove_"):
            row_id = btn_id[len("remove_"):]
            self._remove_entry_row(row_id)
        elif btn_id == "run-btn":
            self._start_run()
        elif btn_id == "clear-log-btn":
            self.action_clear_log()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide inference-specific inputs based on workflow mode."""
        inference_section = self.query_one("#inference-section")
        radio_id = event.pressed.id if event.pressed else ""
        if radio_id == "mode-analysis":
            inference_section.add_class("hidden")
        else:
            inference_section.remove_class("hidden")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_clear_log(self) -> None:
        self.query_one("#log-output", RichLog).clear()

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def _collect_slide_entries(self) -> List[dict]:
        """Parse slide entries from the Data Sources tab."""
        entries = []
        for row_id in self._inference_rows:
            try:
                inp = self.query_one(f"#entry_{row_id}", Input)
                raw = inp.value.strip()
                if not raw:
                    continue
                parts = [p.strip() for p in raw.split("|")]
                slide_id = parts[0] if len(parts) > 0 else ""
                region_coords = parts[1] if len(parts) > 1 else ""
                gt_path = parts[2] if len(parts) > 2 else ""
                entries.append({
                    "slide_id": slide_id,
                    "region_coords": region_coords,
                    "gt_path": gt_path,
                })
            except Exception as exc:
                logger.warning("Failed to parse entry row %s: %s", row_id, exc)
        return entries

    def _collect_model_entries(self) -> List[dict]:
        """Parse model entries from the Models tab."""
        entries = []
        for row_id in self._model_rows:
            try:
                inp = self.query_one(f"#entry_{row_id}", Input)
                raw = inp.value.strip()
                if not raw:
                    continue
                parts = [p.strip() for p in raw.split("|")]
                checkpoint_path = parts[0] if len(parts) > 0 else ""
                label = parts[1] if len(parts) > 1 else Path(checkpoint_path).stem
                entries.append({
                    "checkpoint_path": checkpoint_path,
                    "label": label,
                })
            except Exception as exc:
                logger.warning("Failed to parse entry row %s: %s", row_id, exc)
        return entries

    def _get_workflow_mode(self) -> str:
        """Return 'full', 'infer', or 'analysis'."""
        radio_set = self.query_one("#workflow-radio", RadioSet)
        pressed = radio_set.pressed_button
        if pressed is not None:
            if pressed.id == "mode-infer":
                return "infer"
            if pressed.id == "mode-analysis":
                return "analysis"
        return "full"

    def _start_run(self) -> None:
        mode = self._get_workflow_mode()
        slides = self._collect_slide_entries()
        models = self._collect_model_entries()
        output_dir = self.query_one("#output-dir", Input).value.strip()
        config_file = self.query_one("#config-file", Input).value.strip() or None
        summary_path = self.query_one("#summary-path", Input).value.strip() or None
        split_file = self.query_one("#split-file", Input).value.strip() or None
        device = self.query_one("#device", Input).value.strip() or "cuda"
        scenario = self.query_one("#scenario-select", Select).value

        log_widget = self.query_one("#log-output", RichLog)

        if not output_dir:
            log_widget.write(
                "[bold red]Error: Output Directory is required.[/bold red]"
            )
            self.query_one("#tabs").active = "tab-log"
            return

        # Scenario-specific validation warnings
        if scenario == "single":
            if len(slides) > 1:
                log_widget.write(
                    "[yellow]Warning: Single Run scenario expects 1 slide entry; "
                    f"{len(slides)} configured.[/yellow]"
                )
            if len(models) > 1:
                log_widget.write(
                    "[yellow]Warning: Single Run scenario expects 1 model entry; "
                    f"{len(models)} configured.[/yellow]"
                )
        elif scenario == "batch":
            if len(models) > 1:
                log_widget.write(
                    "[yellow]Warning: Batch Processing scenario expects 1 model entry; "
                    f"{len(models)} configured.[/yellow]"
                )
        elif scenario == "model_compare":
            if len(slides) > 1:
                log_widget.write(
                    "[yellow]Warning: Model Comparison scenario expects 1 slide entry; "
                    f"{len(slides)} configured.[/yellow]"
                )

        self.query_one("#tabs").active = "tab-log"
        self._run_workflow(
            mode=mode,
            scenario=str(scenario),
            slides=slides,
            models=models,
            output_dir=output_dir,
            config_file=config_file,
            summary_path=summary_path,
            split_file=split_file,
            device=device,
        )

    @work(thread=True)
    def _run_workflow(
        self,
        mode: str,
        scenario: str,
        slides: List[dict],
        models: List[dict],
        output_dir: str,
        config_file: Optional[str],
        summary_path: Optional[str],
        split_file: Optional[str],
        device: str,
    ) -> None:
        """Background worker that runs the selected workflow."""
        logger.info("=" * 60)
        logger.info("Starting workflow  mode=%s  scenario=%s", mode, scenario)
        logger.info("  Slides : %d", len(slides))
        logger.info("  Models : %d", len(models))
        logger.info("  Output : %s", output_dir)

        run_dirs: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Inference (Full or Inference Only)
        # ------------------------------------------------------------------
        if mode in ("full", "infer"):
            from histvis.inference import run_inference

            if not slides:
                logger.warning("No slide entries configured – skipping inference.")
            elif not models:
                logger.warning("No model entries configured – skipping inference.")
            else:
                for slide in slides:
                    for model in models:
                        slide_id = slide.get("slide_id", "unknown")
                        region_coords = slide.get("region_coords", "")
                        checkpoint_path = model.get("checkpoint_path", "")
                        model_label = model.get("label", Path(checkpoint_path).stem)
                        sub_dir = str(
                            Path(output_dir)
                            / f"slide_{slide_id}"
                            / f"pred_{model_label}"
                        )
                        try:
                            out = run_inference(
                                slide_id=slide_id,
                                region_coords=region_coords,
                                checkpoint_path=checkpoint_path,
                                output_dir=sub_dir,
                                config_file=config_file,
                                summary_path=summary_path,
                                split_file=split_file,
                                device=device,
                            )
                            run_dirs.append(str(out))
                        except Exception as exc:
                            logger.error(
                                "Inference failed for slide=%s model=%s: %s",
                                slide_id, model_label, exc,
                            )

        # ------------------------------------------------------------------
        # Step 2 (Analysis Only): collect existing .npy directories
        # ------------------------------------------------------------------
        if mode == "analysis":
            base = Path(output_dir)
            if base.exists():
                for candidate in sorted(base.rglob("prediction.npy")):
                    run_dirs.append(str(candidate.parent))
            else:
                logger.error("Output directory does not exist: %s", output_dir)
                return

        if mode in ("full", "analysis"):
            if not run_dirs:
                logger.warning("No run directories found – cannot run analysis.")
            else:
                # Step 2a: Error maps
                try:
                    from histvis.eval_errors import generate_error_maps
                    logger.info("Generating error maps …")
                    generate_error_maps(run_dirs=run_dirs, output_dir=str(Path(output_dir) / "error_maps"))
                except Exception as exc:
                    logger.error("Error map generation failed: %s", exc, exc_info=True)

                # Step 2b: Metrics
                try:
                    from histvis.eval_metrics import calculate_metrics
                    logger.info("Calculating metrics …")
                    df = calculate_metrics(run_dirs=run_dirs, output_dir=str(Path(output_dir) / "metrics"))
                    if not df.empty:
                        logger.info("Metrics summary:\n%s", df.to_string(index=False))
                except Exception as exc:
                    logger.error("Metrics calculation failed: %s", exc, exc_info=True)

                # Step 2c: Scatter plots
                try:
                    from histvis.eval_scatter import generate_scatter_plots
                    logger.info("Generating scatter plots …")
                    generate_scatter_plots(run_dirs=run_dirs, output_dir=str(Path(output_dir) / "scatter"))
                except Exception as exc:
                    logger.error("Scatter plot generation failed: %s", exc, exc_info=True)

        logger.info("Workflow complete.")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point – launched when the user types ``histvis``."""
    app = HistVisApp()
    app.run()


if __name__ == "__main__":
    main()
