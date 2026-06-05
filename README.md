<div align="center">

# Fast-SAM3D **++** &nbsp;·&nbsp; HiCache++

**Image-to-3D, faster — Fast-SAM3D with a tree-aware HiCache++ cache that forecasts the slat-stage
velocity with an *exponential* (DMD / Prony) basis.**

*Replace HiCache's polynomial forecast with a Dynamic-Mode-Decomposition exponential basis — exact
on the feature-ODE class the slat velocities live in, so it stays lossless at larger skip intervals
than the polynomial. Training-free, geometry-preserving, native (no monkey-patching).*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![method HiCache++](https://img.shields.io/badge/cache-HiCache%2B%2B%20(DMD%2FProny)-2e6db0)
&nbsp;![upstream Fast--SAM3D](https://img.shields.io/badge/fork%20of-Fast--SAM3D-555)

</div>

---

## What it is

A fork of [**Fast-SAM3D**](https://github.com/wlfeng0509/Fast-SAM3D) (single-image → textured 3D
mesh, itself a TaylorSeer-style accelerated [SAM3D](https://github.com/facebookresearch/sam-3d-objects))
that adds **HiCache++** — an *exponential* feature cache — to the **slat-stage flow-matching sampler**.

SAM3D generates in two flow-matching stages; the second (**slat**) stage integrates an ODE
`dx/dt = v_θ(x, t)` with a Euler solver, where each `v_θ` call is an expensive DiT forward.
Like HiCache, HiCache++ runs the network on a sparse schedule and *forecasts* the (CFG-combined)
velocity on **skipped** steps instead of calling `v_θ` — but it forecasts with a **DMD/Prony
exponential** basis instead of a polynomial. SAM3D's velocities are `torch.utils._pytree`
structures, so the forecaster is **tree-aware**: each snapshot is flattened to one vector, the DMD
propagator is identified and advanced, and the result is unflattened back to the tree. **HiCache
(Hermite)** is kept in the same module as the warm-up forecaster and the head-to-head baseline; the
companion **Adaptive-CFG** drops the unconditional pass once it aligns with the conditional one.
Wiring is **native** — no runtime patching.

## Method — DMD/Prony exponential velocity forecasting

A flow-matching velocity trajectory is the solution of a slowly-varying, near-linear feature-ODE
`Ḟ = M F`, whose **exact** solution class is a *sum of (damped/oscillatory) exponentials*
`F_t = Σⱼ aⱼ e^{μⱼ t}` — **not** a polynomial. HiCache's scaled-Hermite forecast is only a *local
Taylor truncation* of that exponential, so it diverges as the skip grows; that is precisely what
caps a polynomial cache's lossless interval.

HiCache++ forecasts with **Dynamic Mode Decomposition** (Schmid 2010), the SVD-regularised
generalisation of **Prony's method**: identify the linear propagator `A` from raw velocity
snapshots (`F_{t+1} ≈ A F_t`), eigendecompose it once, and advance any (fractional) horizon `k` by
eigenvalue powers, `F_{t+k} ≈ Φ (λᵏ ⊙ b)`. Because the exponentials **are** the exact solution
class — the property the polynomial lacks — DMD holds quality at skip intervals where Hermite
drifts, and its fractional horizon lets it forecast sub-steps between compute steps exactly. It
needs ≥4 uniform snapshots to fit (a real-valued oscillatory mode costs two real DOF per complex
pole); below that floor it falls back to the Hermite forecast for warm-up.

## Enable it (real API)

The cache attaches to the slat-stage `FlowMatching` generator's Euler solver. The model methods are
chainable (they return the model):

```python
# `fm` is the slat-stage FlowMatching generator inside the Fast-SAM3D pipeline.

# HiCache++ : the DMD/Prony exponential forecaster (Hermite covers the warm-up window).
fm.enable_dmd(
    interval=6,        # run the DiT 1 step in `interval`; forecast the other (interval-1)
    first_enhance=2,   # always run full for the first N steps (warm-up)
    end_enhance=None,  # always run full for the final steps (defaults to the last step)
    history=5,         # snapshots kept for the DMD fit (>=4 to leave Hermite warm-up)
    max_order=2, sigma=0.5,   # Hermite fallback params used until the DMD floor is met
)

# Equivalent, via the unified entry point with an explicit basis:
fm.enable_hicache(interval=6, backend="dmd", history=5)   # backend="hermite" -> plain HiCache

# ... run the pipeline / sampler as usual ...

fm.disable_hicache()   # back to the dense (uncached) schedule
```

Both `enable_dmd` and `enable_hicache(..., backend="dmd")` store the config on the Euler solver;
on each step the solver calls `hicache_decide` and, when it returns `"forecast"`, dispatches to
`dmd_forecast_tree` (DMD) or `hicache_forecast_tree` (Hermite) per `backend`, replacing the
`dynamics_fn` (DiT) call — see
[`accel.py`](sam3d_objects/model/backbone/generator/flow_matching/accel.py),
[`solver.py`](sam3d_objects/model/backbone/generator/flow_matching/solver.py), and
[`model.py`](sam3d_objects/model/backbone/generator/flow_matching/model.py).

## Results

On Fast-SAM3D's slat-stage FlowMatching, **HiCache++ (DMD) is geometry-lossless (F1 = 1.000) out to
interval-6** — the same FlowMatching substrate as SAM3D, where the exponential basis holds quality
**two intervals further** than HiCache's polynomial (Hermite is lossless to interval-3). At
interval-6 it also gives the best speedup of the lossless configs. Full A/B tables, the controlled
forecast microbenchmark, and the Hunyuan3D / SAM3D / Fast-SAM3D numbers are in the standalone
library [`hicache-plus-plus`](../hicache-plus-plus); plain HiCache lives in
[`fastsam3d-plus`](../fastsam3d-plus).

> The basis swap moves latency only on the **slat** stage (where the forecaster replaces DiT
> calls). The SS stage already runs a fixed TaylorSeer stride, so Hermite ⇄ DMD there is a wash.

## Attribution

- **Fast-SAM3D** — © [wlfeng0509](https://github.com/wlfeng0509/Fast-SAM3D)
  ([arXiv:2602.05293](https://arxiv.org/abs/2602.05293)); built on
  [SAM3D](https://github.com/facebookresearch/sam-3d-objects). The upstream README is preserved in
  this fork's git history and its license/attribution is unchanged.
- **HiCache** — scaled-Hermite velocity forecasting, [arXiv:2508.16984](https://arxiv.org/abs/2508.16984)
  (the polynomial baseline retained here as warm-up + comparison).
- **HiCache++** *(this work)* — the **DMD/Prony exponential** velocity forecaster. DMD (Schmid 2010)
  / Prony (1795) are classical spectral estimation; their application to diffusion / flow-matching
  feature caching is, to our knowledge, new. Standalone: [`hicache-plus-plus`](../hicache-plus-plus).
- **Adaptive-CFG** — Adaptive Guidance, [arXiv:2312.12487](https://arxiv.org/abs/2312.12487).

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[wlfeng0509/Fast-SAM3D](https://github.com/wlfeng0509/Fast-SAM3D), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
