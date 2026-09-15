# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Fast-SAM3D's compatibility surface for the central ``hicache-pp`` PyTree API.

The solver deals in structured (PyTree) velocities, so the SLaT flow-matching
path uses :mod:`hicache_pp.tree` directly. Keeping this small module preserves
the imports used by the upstream solver and downstream integrations while
ensuring that scheduling, DMD fitting, snapshot ownership, and telemetry have
one implementation.

The SS stage's native TaylorSeer/carving path is intentionally outside this
module and is not changed by enabling SLaT HiCache or DMD.
"""

try:
    from hicache_pp.tree import (
        adaptive_cfg_decide,
        adaptive_cfg_init,
        dmd_forecast_tree,
        dmd_update_snapshots_tree,
        forecast_guidance_tree,
        guidance_term_tree,
        hermite_coeff,
        hicache_decide,
        hicache_forecast_tree,
        hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_tree,
        physicists_hermite,
        reconstruct_cfg_tree,
        tree_axpy,
        tree_cosine,
        tree_detach,
        tree_sub_div,
    )
except ImportError as exc:  # pragma: no cover - exercised by installation checks
    raise ImportError(
        "Fast-SAM3D++ acceleration requires hicache-pp>=1.2.1; "
        "install the project dependencies before importing flow_matching"
    ) from exc


__all__ = [
    "adaptive_cfg_decide",
    "adaptive_cfg_init",
    "dmd_forecast_tree",
    "dmd_update_snapshots_tree",
    "forecast_guidance_tree",
    "guidance_term_tree",
    "hermite_coeff",
    "hicache_decide",
    "hicache_forecast_tree",
    "hicache_init",
    "hicache_reset",
    "hicache_telemetry",
    "hicache_update_tree",
    "physicists_hermite",
    "reconstruct_cfg_tree",
    "tree_axpy",
    "tree_cosine",
    "tree_detach",
    "tree_sub_div",
]
