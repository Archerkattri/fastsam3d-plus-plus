#!/usr/bin/env python3
"""SAM3D flow-matching (slat-stage) HiCache / HiCache++ (DMD) A/B harness.

The runnable SAM3D pipeline here is Fast-SAM3D's InferencePipelinePointMap; its
``slat_generator`` is a ``sam3d_objects`` FlowMatching — the SAME architecture as the
standalone (gated, undownloaded) sam-3d-objects repo where the HiCache/DMD port also
lives. This toggles HiCache (Hermite) / HiCache++ (DMD) on that FlowMatching's Euler
solver and measures geometry drift (Chamfer / F1 of the output gaussians vs the vanilla
baseline) + latency. A warmup pass keeps latency honest (no cold-start artifact).

Run in the sam3d-objects env (deps bridged from gim_env):
  CUDA_VISIBLE_DEVICES=0 CONDA_PREFIX=$ENVP PATH=$ENVP/bin:$PATH TORCH_HOME=~/.cache/torch \
    $ENVP/bin/python third_party/Fast-SAM3D/sam3d_slat_dmd_ab.py --gpu 0
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

FASTSAM3D_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mask-index", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(FASTSAM3D_ROOT / "ab_slat_out"))
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.chdir(FASTSAM3D_ROOT)
    sys.path.insert(0, str(FASTSAM3D_ROOT))
    sys.path.insert(0, str(FASTSAM3D_ROOT / "notebook"))

    import torch
    # reuse the established pipeline-loader + scorer from the SS harness
    from ab_accel_bench import build_args, build_inference, extract_points, discover_images, M
    import numpy as _np
    from PIL import Image as _Image

    def load_image(p):
        return _np.array(_Image.open(p)).astype(_np.uint8)

    def load_single_mask(folder, index=0, ext=".png"):
        m = _np.array(_Image.open(os.path.join(folder, f"{index}{ext}"))) > 0
        return m[..., -1] if m.ndim == 3 else m

    inference = build_inference("hf", build_args(a.mask_index, a.seed))
    slat = inference._pipeline.models["slat_generator"]
    print(f"slat generator: {type(slat).__name__}  enable_dmd={hasattr(slat, 'enable_dmd')}", flush=True)

    CONFIGS = [
        ("vanilla",      None),
        ("hicache_i3o2", dict(method="hicache", interval=3, max_order=2, first_enhance=3, sigma=0.5)),
        ("dmd_i4",       dict(method="dmd", interval=4, first_enhance=3, history=5, max_order=2, sigma=0.5)),
        ("dmd_i5",       dict(method="dmd", interval=5, first_enhance=4, history=6, max_order=3, sigma=0.5)),
        ("dmd_i6",       dict(method="dmd", interval=6, first_enhance=4, history=6, max_order=3, sigma=0.55)),
    ]

    def apply(cfg):
        if hasattr(slat, "disable_hicache"):
            slat.disable_hicache()
        if cfg is None:
            return
        m = dict(cfg); meth = m.pop("method")
        (slat.enable_dmd if meth == "dmd" else slat.enable_hicache)(**m)

    items = discover_images(None, a.mask_index)
    if not items:
        raise SystemExit("no usable image dirs")
    name, img_path, mfolder, mi = items[0]
    image = load_image(img_path); mask = load_single_mask(mfolder, index=mi)

    def run(cfg):
        apply(cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = inference(image, mask, seed=a.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return extract_points(out), dt

    # warmup (discarded) so the first timed config isn't charged CUDA autotune/compile
    try:
        run(None); print("warmup gen done (discarded)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"warmup skipped: {e!r}", flush=True)

    rows = []; base_pts = None; base_lat = None
    for label, cfg in CONFIGS:
        try:
            pts, dt = run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {label}: {e!r}", flush=True); continue
        if label == "vanilla":
            base_pts, base_lat = pts, dt; cd, f1 = 0.0, 1.0
        else:
            cd, f1, _, _ = M.chamfer_and_f1(pts, base_pts, 0.05)
        sp = (base_lat / dt) if base_lat else 1.0
        rows.append(dict(config=label, lat=round(dt, 3), speedup=round(sp, 3),
                         CD_vs_base=round(float(cd), 4), F1_vs_base=round(float(f1), 3)))
        print(f"  {label:16} {dt:6.2f}s  speedup={sp:.2f}x  CD={float(cd):.4f}  F1={float(f1):.3f}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\nconfig            lat     speedup  CD_vs_base  F1_vs_base")
    for r in rows:
        print(f"{r['config']:16} {r['lat']:6.2f}  {r['speedup']:6.2f}x  {r['CD_vs_base']:10.4f}  {r['F1_vs_base']:10.3f}")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
