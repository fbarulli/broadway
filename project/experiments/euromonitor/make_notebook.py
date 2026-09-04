"""make_notebook.py — thin builder: notebook.md -> entity_resolution.ipynb.

The notebook CONTENT lives in ``notebook.md`` as plain text — edit that file to
rewrite the notebook. Cells are separated by one-line markers: ``%%md`` starts a
markdown cell, ``%%code`` starts a code cell. This file only:

  1. parses notebook.md into cells,
  2. executes the code cells in-process (baking figures/tables as outputs),
  3. writes entity_resolution.ipynb.

The helper functions below are imported by the notebook's code cells — declare
any new plot/helper here, call it from a ``%%code`` block in notebook.md.

Regenerate with:  python project/experiments/euromonitor/make_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # euromonitor dir -> _common
from _common import RESULTS, load_dataset, load_dataset_deduped

# ---------------------------------------------------------------------------
# Helpers — imported by the notebook's %%code cells, declared here.
# ---------------------------------------------------------------------------


def plot_dataset_sizes() -> None:
    """2-bar chart: original vs deduplicated dataset size (entries)."""
    import matplotlib.pyplot as plt

    n_raw = len(load_dataset())
    n_dedup = len(load_dataset_deduped())
    removed = n_raw - n_dedup
    pct = removed / n_raw * 100

    _fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(
        ["Original dataset", "Deduplicated dataset"],
        [n_raw, n_dedup],
        width=0.55,
        color=["#4C72B0", "#55A868"],
    )
    for bar, n in zip(bars, (n_raw, n_dedup), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, n, f"{n:,}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )
    ax.set_ylabel("entries")
    ax.set_ylim(0, n_raw * 1.22)
    ax.set_title(f"{removed:,} duplicate listings removed (−{pct:.1f}%)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def show_image(name: str) -> None:
    """Display a PNG from the results dir (e.g. ``01g_usable_by_country.png``)."""
    import matplotlib.pyplot as plt

    img = plt.imread(str(RESULTS / name))
    h, w = img.shape[:2]
    _fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(img)
    ax.axis("off")
    plt.show()


def show_image_row(names: list[str], *, height_in: float = 5.0) -> None:
    """Display several results PNGs side by side with one common height."""
    import matplotlib.pyplot as plt

    imgs = [plt.imread(str(RESULTS / n)) for n in names]
    aspects = [im.shape[1] / im.shape[0] for im in imgs]
    widths = [height_in * a for a in aspects]
    gap = 0.35 * (len(names) - 1)
    _fig, axes = plt.subplots(1, len(names), figsize=(sum(widths) + gap, height_in))
    if len(names) == 1:
        axes = [axes]
    for ax, im, name in zip(axes, imgs, names, strict=True):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(name, fontsize=9)
    plt.show()


def plot_barcode_audit() -> None:
    """Donut: clean vs conflicting barcodes among identical cross-retailer titles."""
    import matplotlib.pyplot as plt

    s = pd.read_csv(RESULTS / "06b_conflicting_summary.csv").set_index("metric")["value"]
    clean = int(s["clean_calibration_positives"])
    conflicts = int(s["conflicting_barcode_groups"])
    total = clean + conflicts
    _fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.pie(
        [clean, conflicts],
        labels=[f"Clean calibration positives\n{clean:,} ({clean / total:.1%})",
                f"Conflicting barcodes\n{conflicts:,} ({conflicts / total:.1%}) — flagged for review"],
        colors=["#4C72B0", "#C44E52"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.4},  # donut hole
        textprops={"fontsize": 9},
    )
    ax.set_title("Identical titles across retailers: barcode agreement")
    plt.show()


# ---------------------------------------------------------------------------
# Builder — run as a script; never imported by the notebook.
# ---------------------------------------------------------------------------


def _read_cells(path: Path) -> list[dict]:
    """Parse notebook.md (``%%md`` / ``%%code`` markers) into notebook cells."""
    cells: list[dict] = []
    current: dict | None = None
    buffer: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("%%"):
            if current is not None:
                current["source"] = "\n".join(buffer).strip("\n") + "\n"
                cells.append(current)
            kind = line[2:].strip()
            if kind == "md":
                current = {"cell_type": "markdown", "metadata": {}, "source": ""}
            elif kind == "code":
                current = {"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": ""}
            else:
                raise ValueError(f"unknown cell marker: {line!r}")
            buffer = []
        else:
            if current is None:
                raise ValueError(f"content before the first %% marker: {line!r}")
            buffer.append(line)
    if current is not None:
        current["source"] = "\n".join(buffer).strip("\n") + "\n"
        cells.append(current)
    return cells


def _build() -> None:
    import base64
    import contextlib
    import io
    import json
    import os
    import re
    import traceback

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell

    EURO = Path(__file__).resolve().parent
    CONTENT = EURO / "notebook.md"
    NOTEBOOK = EURO / "entity_resolution.ipynb"
    os.chdir(EURO)

    cells = _read_cells(CONTENT)
    shell = InteractiveShell.instance()

    def run_cell(src: str):
        out = io.StringIO()
        images = []
        orig_show = plt.show

        def fake_show(*a, **k):
            for num in plt.get_fignums():
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
                images.append(buf.getvalue())
                buf.close()
            plt.close("all")

        plt.show = fake_show
        err = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                res = shell.run_cell(src, store_history=False)
            err = res.error_in_exec
        finally:
            plt.show = orig_show
            plt.close("all")

        outputs = []
        text = re.sub(r"^Out\[\d+\]:\s*\n", "", out.getvalue(), flags=re.MULTILINE)
        if text:
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True) or ["\n"]})
        for png in images:
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png).decode("ascii"),
                                     "text/plain": ["<Figure>"]}})
        if err is not None:
            outputs.append({"output_type": "error", "ename": type(err).__name__,
                            "evalue": str(err),
                            "traceback": traceback.format_exception(type(err), err, err.__traceback__)})
        return outputs, err

    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], err = run_cell(c["source"])
            if err is not None:
                print(f"CELL ERROR: {type(err).__name__}: {err}", file=sys.stderr)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK} with {len(cells)} cells")


if __name__ == "__main__":
    _build()
