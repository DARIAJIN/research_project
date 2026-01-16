from pathlib import Path
import sys
import time

import napari
import nibabel as nib
import numpy as np

from magicgui import magicgui
from napari.qt.threading import thread_worker

from qtpy.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout

# ============================================================
# Project root
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

from rsa_planning_gui.icp_pipeline.full_pipeline import run_full_pipeline

# ============================================================
# Demo input
# ============================================================

COMPLETE_DIR = PROJECT_DIR / "demo" / "segmentations"
assert COMPLETE_DIR.exists(), f"Complete data not found: {COMPLETE_DIR}"

GUI_WORK_DIR = PROJECT_DIR / "demo" / "_gui_tmp"
GUI_WORK_DIR.mkdir(parents=True, exist_ok=True)

STRUCT = "scapula"

# ============================================================
# Utils
# ============================================================

def mask_to_rgba(mask, color, alpha=0.4):
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask.astype(float) * alpha
    return rgba


# ============================================================
# napari viewer
# ============================================================

viewer = napari.Viewer(title="RSA ICP Demo (Idle)")

# ============================================================
# GUI log widget
# ============================================================

class LogWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)

    def log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        self.text.appendPlainText(line)
        print(line)   # 同时打印到终端


log_widget = LogWidget()
viewer.window.add_dock_widget(log_widget, name="Pipeline Log", area="right")

def log(msg):
    log_widget.log(msg)

# ============================================================
# Background worker
# ============================================================

@thread_worker
def full_pipeline_worker():
    log("PIPELINE: start full pipeline")
    fixed, before, after = run_full_pipeline(
        complete_dir=COMPLETE_DIR,
        work_dir=GUI_WORK_DIR,
        struct=STRUCT
    )
    log("PIPELINE: finished successfully")
    return fixed, before, after

# ============================================================
# Visualization callback
# ============================================================

def show_results(result):
    log("GUI: updating visualization")
    viewer.title = "RSA ICP Demo (Finished)"

    fixed_path, before_path, after_path = result

    fixed  = nib.load(str(fixed_path)).get_fdata() > 0
    before = nib.load(str(before_path)).get_fdata() > 0
    after  = nib.load(str(after_path)).get_fdata() > 0

    viewer.layers.clear()

    viewer.add_image(
        mask_to_rgba(fixed, color=(0, 0, 1)),
        name="Fixed (Right)",
        blending="additive"
    )

    viewer.add_image(
        mask_to_rgba(before, color=(0, 1, 0)),
        name="Moving (Left, Before)",
        blending="additive"
    )

    viewer.add_image(
        mask_to_rgba(after, color=(1, 0, 0)),
        name="Moving (After ICP)",
        blending="additive"
    )

    log("GUI: visualization updated")

# ============================================================
# GUI button
# ============================================================

@magicgui(call_button="Run FULL ICP Pipeline")
def run_pipeline_gui():
    log("GUI: Run button clicked")
    viewer.title = "RSA ICP Demo (Running...)"

    worker = full_pipeline_worker()
    worker.returned.connect(show_results)
    worker.start()

viewer.window.add_dock_widget(
    run_pipeline_gui,
    name="ICP Registration",
    area="right"
)

# ============================================================
# Start GUI
# ============================================================

log("GUI ready. Click 'Run FULL ICP Pipeline' to start.")
napari.run()
