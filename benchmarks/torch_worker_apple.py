"""PyTorch reference timing on Apple Silicon (CPU or MPS), for bench_apple.py.

Loads the net once, preprocesses once outside the timed loop, one warmup, then times N forward
passes. Prints a `bench:` line in the same format as da3-cli --repeat.

  python benchmarks/torch_worker_apple.py --model da3 --device mps --image assets/samples/street.jpg

Where the references live (environment variables, defaults under benchmarks/ref/):
  DA3_SRC      a clone of https://github.com/ByteDance-Seed/Depth-Anything-3
  DA3_WEIGHTS  the depth-anything/DA3-SMALL checkpoint (config.json + model.safetensors)
  DA2_WEIGHTS  the depth-anything/Depth-Anything-V2-Small-hf checkpoint
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(HERE, "ref")
DA3_SRC = os.environ.get("DA3_SRC", os.path.join(REF, "da3-src"))
DA3_WEIGHTS = os.environ.get("DA3_WEIGHTS", os.path.join(REF, "DA3-SMALL"))
DA2_WEIGHTS = os.environ.get("DA2_WEIGHTS", os.path.join(REF, "DA2-SMALL-hf"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(image, w, h):
    arr = np.asarray(Image.open(image).convert("RGB").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    return torch.from_numpy(((arr - MEAN) / STD).transpose(2, 0, 1).copy())[None]


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["da3", "da2"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image", required=True)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)

    t0 = time.perf_counter()
    if a.model == "da3":
        sys.path.insert(0, os.path.join(DA3_SRC, "src"))
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from da3_reference import load_model
        _, net = load_model(DA3_WEIGHTS)
        net = net.to(dev).eval()
        bb = net.backbone.pretrained
        W, H = 504, 336  # the C++ production path for a 1024x680 photo
        x = prep(a.image, W, H)[None].to(dev)  # (1,1,3,H,W)

        def forward():
            with torch.no_grad():
                outs, _ = bb.get_intermediate_layers(x, n=[5, 7, 9, 11], export_feat_layers=[],
                                                     ref_view_strategy="saddle_balanced")
                return net.head(list(outs), H, W, patch_start_idx=0)["depth"]
    else:
        from transformers import AutoModelForDepthEstimation
        net = AutoModelForDepthEstimation.from_pretrained(DA2_WEIGHTS).to(dev).eval()
        W, H = 784, 518  # DA2: short side 518, both sides multiples of 14 (the C++ size for this photo)
        x = prep(a.image, W, H).to(dev)

        def forward():
            with torch.no_grad():
                return net(pixel_values=x).predicted_depth
    sync(dev)
    load_ms = (time.perf_counter() - t0) * 1000

    forward()
    sync(dev)
    ms = []
    for _ in range(a.repeat):
        s = time.perf_counter()
        forward()
        sync(dev)
        ms.append((time.perf_counter() - s) * 1000)
    ms.sort()
    print(f"bench: out={W}x{H} device={a.device} load={load_ms:.1f}ms infer={ms[len(ms) // 2]:.1f}ms/iter "
          f"(median over {a.repeat}, min={ms[0]:.1f} max={ms[-1]:.1f} p90={ms[int(len(ms) * 0.9) - 1]:.1f})")


if __name__ == "__main__":
    main()
