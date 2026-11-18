#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yossi Elkrief
"""Apple Silicon benchmark: the original port vs this version (CPU and Metal) vs PyTorch (CPU and MPS).

Each config runs in its own process under `/usr/bin/time -l` (peak RSS), loads the model once,
warms up, then runs `--repeat` forwards. Reported per config:
  load_ms      model load (weights read + backend upload)
  forward_ms   median backbone+head graph time (C++: DA_PROFILE `graph=`; PyTorch: timed forward)
  e2e_ms       median end-to-end per image for C++ (decode + preprocess + forward + readback)
  peak_mb      peak resident memory of the process (on unified memory this includes GPU buffers)

  python benchmarks/bench_apple.py --upstream <checkout-of-the-original> --torch-python <venv python> \\
      --torch-worker <torch_worker.py> --out benchmarks/results_apple.json

Run it on a quiet machine: no simulators, emulators, builds or browsers working in the background.
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = os.path.join(ROOT, "assets", "samples", "street.jpg")  # 1024x680 -> 504x336, the production path
MODELS = {
    "da3-small-f32": os.path.join(ROOT, "models", "depth-anything-small-f32.gguf"),
    "da3-small-q8_0": os.path.join(ROOT, "models", "depth-anything-small-q8_0.gguf"),
    "da2-small-q8_0": os.path.join(ROOT, "models", "depth-anything2-small-q8_0.gguf"),
}
BENCH_RE = re.compile(r"load=([\d.]+)ms\s+infer=([\d.]+)ms/iter")
GRAPH_RE = re.compile(r"graph\(backbone\+head\)=([\d.]+)ms")
RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


def run(cmd, env=None, cooldown=5.0):
    time.sleep(cooldown)  # comparable thermal state for every config
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{out[-2000:]}")
    rss = RSS_RE.search(out)
    return out, (int(rss.group(1)) / 2**20 if rss else float("nan"))


def cpp(cli, model, threads, repeat, device):
    env = {"DA_PROFILE": "1"}
    if device == "cpu":
        env["DA_DEVICE"] = "cpu"
    out, rss = run([cli, "depth", "--model", MODELS[model], "--input", IMAGE, "--repeat", str(repeat),
                    "--threads", str(threads)], env)
    load, e2e = map(float, BENCH_RE.search(out).groups())
    graphs = [float(g) for g in GRAPH_RE.findall(out)][1:]  # drop the warmup
    return {"load_ms": load, "e2e_ms": e2e, "forward_ms": statistics.median(graphs) if graphs else None, "peak_mb": rss}


def torch(python, worker, model, device, threads, repeat):
    out, rss = run([python, worker, "--model", model, "--device", device, "--image", IMAGE,
                    "--repeat", str(repeat), "--threads", str(threads)])
    load, fwd = map(float, BENCH_RE.search(out).groups())
    return {"load_ms": load, "forward_ms": fwd, "e2e_ms": None, "peak_mb": rss}


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="checkout of the original port with build-cpu/ and build-metal/")
    ap.add_argument("--fork-cpu", default=os.path.join(ROOT, "build-cpu-fork", "examples", "cli", "da3-cli"))
    ap.add_argument("--fork-metal", default=os.path.join(ROOT, "build-metal-pre", "examples", "cli", "da3-cli"))
    ap.add_argument("--torch-python", required=True)
    ap.add_argument("--torch-worker", default=os.path.join(ROOT, "benchmarks", "torch_worker_apple.py"))
    ap.add_argument("--threads", type=int, default=int(subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout or 8))
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results_apple.json"))
    a = ap.parse_args()

    up_cpu = os.path.join(a.upstream, "build-cpu", "examples", "cli", "da3-cli")
    up_metal = os.path.join(a.upstream, "build-metal", "examples", "cli", "da3-cli")
    rows = []

    def add(engine, device, model, fn):
        r = fn()
        r.update(engine=engine, device=device, model=model)
        rows.append(r)
        print(f"{engine:10s} {device:5s} {model:16s} load {r['load_ms']:7.1f}  forward {r['forward_ms'] or 0:7.1f}  "
              f"e2e {r['e2e_ms'] or 0:7.1f}  peak {r['peak_mb']:7.0f} MB", flush=True)

    for model in MODELS:
        for device, (up, fk) in {"cpu": (up_cpu, a.fork_cpu), "metal": (up_metal, a.fork_metal)}.items():
            add("original", device, model, lambda: cpp(up, model, a.threads, a.repeat, device))
            add("fork", device, model, lambda: cpp(fk, model, a.threads, a.repeat, device))
    for model, tmodel in [("da3-small-f32", "da3"), ("da2-small-f32", "da2")]:
        for device in ["cpu", "mps"]:
            add("pytorch", device, model, lambda: torch(a.torch_python, a.torch_worker, tmodel, device, a.threads, a.repeat))

    meta = {"chip": chip(), "threads": a.threads, "repeat": a.repeat, "image": os.path.basename(IMAGE),
            "macos": platform.mac_ver()[0], "date": time.strftime("%Y-%m-%d")}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
