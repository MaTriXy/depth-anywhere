#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yossi Elkrief
"""Plots for the Apple Silicon and phone benchmarks (results_apple.json, results_mobile.json).

    python benchmarks/make_plots_apple.py   ->  benchmarks/media/{mac_speed,mac_memory,cold_start,phones}.png

Same look as make_plots.py. Colours: this version = blue, PyTorch = orange, the original port = aqua
(a palette checked for colour-blind separation); processing sizes use one blue ramp.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
FORK, TORCH, ORIGINAL = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"504": "#104281", "336": "#2a78d6", "252": "#86b6ef"}
GRID, INK, MUTED = "#e3e3e0", "#222222", "#666666"


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors=INK)


def labels(ax, rects, fmt="{:.0f}"):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=INK)


def bar(ax, x, h, w, color, label):
    # A 2px white edge keeps neighbouring bars apart
    return ax.bar(x, h, w, color=color, label=label, zorder=3, edgecolor="white", linewidth=2)


def save(fig, name):
    path = os.path.join(MEDIA, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def pick(rows, engine, device, model):
    for r in rows:
        if r["engine"] == engine and r["device"] == device and r["model"] == model:
            return r
    raise KeyError((engine, device, model))


def main():
    apple = json.load(open(os.path.join(HERE, "results_apple.json")))
    mobile = json.load(open(os.path.join(HERE, "results_mobile.json")))
    rows, meta = apple["rows"], apple["meta"]
    chip = meta["chip"].replace("Apple ", "")
    m = "da3-small-f32"

    # Forward time, CPU and GPU
    groups = [("CPU", "cpu", "cpu"), ("GPU (Metal / MPS)", "metal", "mps")]
    series = [("Original port", ORIGINAL, "original"), ("This version", FORK, "fork"), ("PyTorch", TORCH, "pytorch")]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x, w = np.arange(len(groups)), 0.26
    for i, (name, color, engine) in enumerate(series):
        vals = [pick(rows, engine, cdev if engine != "pytorch" else tdev, m)["forward_ms"] for _, cdev, tdev in groups]
        labels(ax, bar(ax, x + (i - 1) * w, vals, w, color, name))
    ax.set_xticks(x, [g[0] for g in groups], fontsize=11)
    style(ax, f"DA3-Small forward pass on {chip}", "ms per image (lower is better)")
    ax.legend(frameon=False, fontsize=10)
    fig.text(0.5, -0.03, f"504x336, f32, {meta['threads']} threads on CPU, median of {meta['repeat']}. "
             "Forward = backbone + head, the part PyTorch times.", ha="center", fontsize=9, color=MUTED)
    save(fig, "mac_speed.png")

    # Peak memory
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, (name, color, engine) in enumerate(series):
        vals = [pick(rows, engine, cdev if engine != "pytorch" else tdev, m)["peak_mb"] for _, cdev, tdev in groups]
        labels(ax, bar(ax, x + (i - 1) * w, vals, w, color, name))
    ax.set_xticks(x, [g[0] for g in groups], fontsize=11)
    style(ax, f"Peak memory, DA3-Small on {chip}", "MB, whole process (lower is better)")
    ax.legend(frameon=False, fontsize=10)
    fig.text(0.5, -0.03, "Maximum resident set size; on unified memory it includes GPU buffers. "
             "q8_0 brings ggml to 183 MB (CPU) and 210 MB (Metal).", ha="center", fontsize=9, color=MUTED)
    save(fig, "mac_memory.png")

    # First-launch model load, log scale
    cs = mobile["first_launch_load_ms"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x, w = np.arange(len(cs)), 0.36
    labels(ax, bar(ax, x - w / 2, [c["before"] for c in cs], w, ORIGINAL, "Shaders compiled at launch (before)"),
           "{:,.0f} ms")
    labels(ax, bar(ax, x + w / 2, [c["after"] for c in cs], w, FORK, "Precompiled (this version)"), "{:,.0f} ms")
    ax.set_yscale("log")
    ax.set_ylim(30, 30000)
    ax.set_xticks(x, [c["device"] for c in cs], fontsize=11)
    style(ax, "Model load on first launch (Metal)", "ms, log scale (lower is better)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(0.5, -0.03, "ggml compiles its Metal source the first time an app runs; this version ships the kernels compiled.",
             ha="center", fontsize=9, color=MUTED)
    save(fig, "cold_start.png")

    # Phones: latency per processing size, log scale
    lat = mobile["latency_ms"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x, w = np.arange(len(lat)), 0.26
    for i, size in enumerate(["504", "336", "252"]):
        labels(ax, bar(ax, x + (i - 1) * w, [d[size] for d in lat], w, RAMP[size], f"{size} px"))
    ax.set_yscale("log")
    ax.set_ylim(20, 8000)
    ax.axhline(1000 / 15, ls="--", lw=1.2, color=MUTED, zorder=2)
    ax.text(0.5, 1000 / 15 * 1.06, "15/s", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(x, [f"{d['device']}\n{d['chip']} · {d['backend']}" for d in lat], fontsize=9.5)
    style(ax, "DA3-Small on phones, by processing size", "ms per image, log scale (lower is better)")
    ax.legend(title="longest side", frameon=False, fontsize=10, title_fontsize=9.5)
    fig.text(0.5, -0.06, "q8_0, end to end from an image in memory. 252 px suits live camera depth.",
             ha="center", fontsize=9, color=MUTED)
    save(fig, "phones.png")


if __name__ == "__main__":
    main()
