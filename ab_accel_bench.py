#!/usr/bin/env python3
"""Fast-SAM3D acceleration A/B harness — TaylorSeer vs HiCache (+ Adaptive-CFG).

Runs the SAME Fast-SAM3D pipeline (TaylorSeer-variant config, so our pluggable
forecast basis is the active forecast) under 4 configs and measures
geometry-preservation (deviation from the TaylorSeer baseline) + latency/speedup:

    basis    adacfg   meaning
    taylor   off      BASELINE — original Fast-SAM3D TaylorSeer (monomial)
    hermite  off      HiCache  — faster-trellis / hermit-trellis2 forecast
    taylor   on       + Adaptive-CFG (skip the uncond pass on aligned steps)
    hermite  on       full stack (HiCache + Adaptive-CFG)

  * HiCache       : engaged via GF_FORECAST_BASIS=hermite, read per-forecast by
                    forecast_basis.basis_term (so it switches with no reload).
  * Adaptive-CFG  : toggled on the generators' reverse_fn
                    (.enable_adaptive_guidance / .disable_adaptive_guidance);
                    only fires where the base ClassifierFreeGuidance.inner_forward
                    runs (the SLaT generator; the SS generator's PointmapCFG 3-way
                    path is not wired — so adacfg here measures the SLaT-stage win).
  * Carving       : always on — native to Fast-SAM3D (token_slat/), not toggled.

ONE model load; the TaylorSeer cache + adacfg state re-init per generation, so the
four configs are independent. The acceleration methods are *geometry-preserving*, so
the right A/B is: how far does each config's mesh drift from the baseline mesh, and
how much faster is it. Same model + same input => same canonical frame => no
alignment needed; score in a shared unit cube (symmetric Chamfer + F1@0.05).

GPU required. Run once the GPU is free (e.g. after the Toys4K bench finishes):

    python third_party/Fast-SAM3D/ab_accel_bench.py --tag hf --mask-index 14 --gpu 0
    # add more inputs:  --images notebook/images/<dir1> notebook/images/<dir2> ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

FASTSAM3D_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FASTSAM3D_ROOT.parents[1]                 # third_party/Fast-SAM3D -> gaussianfeels

# the established F@0.05 metric (numpy/scipy only)
sys.path.insert(0, str(REPO_ROOT / "third_party" / "benchmark"))
import metrics as M  # noqa: E402

# the 4 configs and the baseline they are scored against
CONFIGS = [("taylor", False), ("hermite", False), ("taylor", True), ("hermite", True)]
BASELINE = ("taylor", False)


def _label(basis: str, adacfg: bool) -> str:
    return f"{basis}{'+adacfg' if adacfg else ''}"


# ──────────────────────────── build the pipeline (mirrors notebook/infer.py) ──
def build_args(mask_index: int, seed: int) -> Namespace:
    """The argparse Namespace notebook/infer.py builds, with enable_taylor=True so
    the TaylorSeer generator configs (where our pluggable basis lives) are used."""
    return Namespace(
        tag="hf", image_path="", mask_index=mask_index, output_dir="", seed=seed,
        ss_faster_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=0.5,
        slat_thresh=0.5, slat_warmup=2, slat_token_ratio=0.15,
        mesh_spectral_threshold_low=0.5, mesh_spectral_threshold_high=0.7,
        enable_ss_faster=False, enable_slat_token=False, enable_mesh_aggregation=False,
        enable_acceleration=False, enable_taylor=True, enable_easy=False,
    )


def build_inference(tag: str, args_ns: Namespace):
    from omegaconf import OmegaConf
    sys.path.append("notebook")
    from inference import Inference  # noqa: E402

    config_path = f"checkpoints/{tag}/pipeline.yaml"
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    # TaylorSeer variant -> taylor_formula (= our pluggable forecast basis) is the forecast
    config["ss_generator_config_path"] = "ss_generator_taylorseer.yaml"
    config["slat_generator_config_path"] = "slat_generator_taylorseer.yaml"
    inf = Inference(config, compile=False, args=args_ns)
    if hasattr(inf, "get_params"):
        inf.get_params(args_ns)
    return inf


def _generators(inference):
    """The CFG dynamics (reverse_fn) of the SS + SLaT generators, if reachable."""
    pipe = getattr(inference, "_pipeline", None)
    models = getattr(pipe, "models", {}) if pipe is not None else {}
    out = []
    for name in ("ss_generator", "slat_generator"):
        g = models.get(name) if isinstance(models, dict) else None
        rf = getattr(g, "reverse_fn", None) if g is not None else None
        if rf is not None:
            out.append(rf)
    return out


def set_adacfg(inference, on: bool, gamma_bar=0.94, warmup=2, max_order=1) -> int:
    n = 0
    for rf in _generators(inference):
        if on and hasattr(rf, "enable_adaptive_guidance"):
            rf.enable_adaptive_guidance(gamma_bar=gamma_bar, warmup=warmup, max_order=max_order); n += 1
        elif hasattr(rf, "disable_adaptive_guidance"):
            rf.disable_adaptive_guidance()
    return n


# ──────────────────────────── run one config ─────────────────────────────────
def extract_points(output, n: int = 30000) -> np.ndarray:
    """Unit-cube-normalised surface points: prefer the mesh (glb), else splat xyz."""
    glb = output.get("glb") if isinstance(output, dict) else None
    if glb is not None and getattr(glb, "vertices", None) is not None and len(glb.vertices):
        v = np.asarray(glb.vertices, np.float64)
        f = np.asarray(glb.faces, np.int64)
        return M.sample_surface(M.normalize_to_unit_cube(v), f, n, seed=0)
    gs = output.get("gs") if isinstance(output, dict) else None
    if gs is not None and hasattr(gs, "_xyz"):
        xyz = gs._xyz.detach().cpu().numpy().astype(np.float64)
        return M.normalize_to_unit_cube(xyz)
    raise RuntimeError("output has neither a usable 'glb' mesh nor a 'gs' splat")


def run_once(inference, image, mask, seed: int, basis: str, adacfg: bool):
    import torch
    os.environ["GF_FORECAST_BASIS"] = basis
    set_adacfg(inference, adacfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = inference(image, mask, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return extract_points(output), dt


# ──────────────────────────── inputs ─────────────────────────────────────────
def discover_images(images_arg, mask_index: int):
    """[(name, image_path, mask_folder, mask_index), ...]. Each dir holds image.png
    + <idx>.png masks (falls back to the first available mask if idx is absent)."""
    dirs = images_arg or sorted(
        str(p.parent) for p in (FASTSAM3D_ROOT / "notebook" / "images").glob("*/image.png"))
    out = []
    for d in dirs:
        d = Path(d)
        img = d / "image.png"
        if not img.exists():
            print(f"  [skip] {d}: no image.png", flush=True)
            continue
        mi = mask_index
        if not (d / f"{mi}.png").exists():
            masks = sorted(int(p.stem) for p in d.glob("*.png")
                           if p.stem.isdigit() and p.name != "image.png")
            if not masks:
                print(f"  [skip] {d}: no <idx>.png mask", flush=True)
                continue
            mi = masks[0]
        out.append((d.name, str(img), str(d), mi))
    return out


# ──────────────────────────── main ───────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--images", nargs="*", default=None, help="image dirs (default: notebook/images/*)")
    ap.add_argument("--mask-index", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gamma-bar", type=float, default=0.94)
    ap.add_argument("--out", default=str(FASTSAM3D_ROOT / "ab_accel_out"))
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.chdir(FASTSAM3D_ROOT)                              # checkpoints/ + notebook/ are relative
    sys.path.insert(0, str(FASTSAM3D_ROOT))               # forecast_basis, taylor_utils_*
    sys.path.insert(0, str(FASTSAM3D_ROOT / "notebook"))  # inference.py lives here
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # Inline the two trivial loaders instead of `from inference import ...`, which would
    # drag in notebook/inference.py's gradio/seaborn/matplotlib demo deps we don't need.
    import numpy as _np
    from PIL import Image as _Image
    def load_image(path):
        return _np.array(_Image.open(path)).astype(_np.uint8)
    def load_single_mask(folder_path, index=0, extension=".png"):
        m = _np.array(_Image.open(os.path.join(folder_path, f"{index}{extension}"))) > 0
        return m[..., -1] if m.ndim == 3 else m

    items = discover_images(a.images, a.mask_index)
    if not items:
        raise SystemExit("no usable image dirs (need <dir>/image.png + <idx>.png mask)")
    print(f"images={len(items)}  configs={[_label(*c) for c in CONFIGS]}", flush=True)

    inference = build_inference(a.tag, build_args(a.mask_index, a.seed))
    nrf = set_adacfg(inference, True, gamma_bar=a.gamma_bar); set_adacfg(inference, False)
    print(f"adaptive-guidance reachable on {nrf} generator reverse_fn(s)", flush=True)

    # Warmup (discarded): the first generation pays CUDA autotune/kernel-compile cost, which
    # would otherwise be charged entirely to whichever config runs first (the baseline),
    # producing a bogus N-x "speedup" for the rest. Run one throwaway gen so all timed runs
    # are warm and the latency comparison is honest.
    try:
        _wimg = load_image(items[0][1]); _wmask = load_single_mask(items[0][2], index=items[0][3])
        run_once(inference, _wimg, _wmask, a.seed, "taylor", False)
        print("warmup gen done (discarded)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"warmup skipped: {e!r}", flush=True)

    rows = []
    for name, img_path, mfolder, mi in items:
        image = load_image(img_path)
        mask = load_single_mask(mfolder, index=mi)
        base_pts = None
        for basis, adacfg in CONFIGS:
            try:
                pts, dt = run_once(inference, image, mask, a.seed, basis, adacfg)
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {name} {_label(basis,adacfg)}: {e!r}", flush=True)
                continue
            if (basis, adacfg) == BASELINE:
                base_pts = pts
                cd = 0.0; f1 = 1.0
            elif base_pts is not None:
                cd, f1, _, _ = M.chamfer_and_f1(pts, base_pts, 0.05)
            else:
                cd = float("nan"); f1 = float("nan")
            row = dict(object=name, basis=basis, adacfg=adacfg, config=_label(basis, adacfg),
                       latency_s=round(dt, 3), CD_vs_base=round(float(cd), 5),
                       F1_vs_base=round(float(f1), 4), n_pts=int(len(pts)))
            rows.append(row)
            print(f"  {name:28} {_label(basis,adacfg):16} {dt:6.2f}s  "
                  f"CDvb={row['CD_vs_base']:.4f} F1vb={row['F1_vs_base']:.3f}", flush=True)

    (out / "cells.json").write_text(json.dumps(rows, indent=2))

    # aggregate: per-config median latency + speedup vs baseline + mean drift
    import statistics
    base_lat = statistics.median([r["latency_s"] for r in rows
                                  if (r["basis"], r["adacfg"]) == BASELINE] or [float("nan")])
    print(f"\n{'config':16} {'n':>3} {'med_lat':>8} {'speedup':>8} {'CD_vs_base':>11} {'F1_vs_base':>11}")
    summ = {}
    for basis, adacfg in CONFIGS:
        rs = [r for r in rows if r["basis"] == basis and r["adacfg"] == adacfg]
        if not rs:
            continue
        lat = statistics.median([r["latency_s"] for r in rs])
        cdv = statistics.mean([r["CD_vs_base"] for r in rs])
        f1v = statistics.mean([r["F1_vs_base"] for r in rs])
        spd = (base_lat / lat) if lat else float("nan")
        summ[_label(basis, adacfg)] = dict(n=len(rs), median_latency_s=round(lat, 3),
                                            speedup_vs_baseline=round(spd, 3),
                                            mean_CD_vs_base=round(cdv, 5), mean_F1_vs_base=round(f1v, 4))
        print(f"{_label(basis,adacfg):16} {len(rs):>3} {lat:8.2f} {spd:8.3f}x "
              f"{cdv:11.4f} {f1v:11.3f}")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nwrote {out}/summary.json  (+ cells.json)")
    print("Read: HiCache (hermite) should match the baseline geometry (CD_vs_base ~0, "
          "F1_vs_base ~1) while being >= as fast; that's the win.")


if __name__ == "__main__":
    main()
