"""CPU contract tests for the Fast-SAM3D++ PyTree adapter."""

import importlib.util
from pathlib import Path

import hicache_pp.tree as central
import pytest
import torch

from sam3d_objects.model.backbone.generator.flow_matching import accel
from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    Euler,
    RungeKutta4,
)


def _tree(value):
    return {"x": torch.tensor([value, value + 1.0]), "nested": {"y": torch.tensor([[value]])}}


def test_accel_is_a_facade_over_central_tree_api():
    assert accel.hicache_init is central.hicache_init
    assert accel.dmd_forecast_tree is central.dmd_forecast_tree
    assert accel.hicache_telemetry is central.hicache_telemetry


def test_central_snapshots_own_their_tensor_storage():
    state = central.hicache_init(num_steps=8, first_enhance=0)
    state["activated_steps"].append(0)
    source = _tree(1.0)
    central.hicache_update_tree(state, source)
    central.dmd_update_snapshots_tree(state, source)

    source["x"].fill_(99.0)
    source["nested"]["y"].fill_(99.0)

    assert torch.equal(state["derivatives"][0]["x"], torch.tensor([1.0, 2.0]))
    assert torch.equal(state["dmd_snapshots"][0][1]["nested"]["y"], torch.tensor([[1.0]]))


def test_dmd_fit_is_reused_until_a_new_compute_anchor(monkeypatch):
    state = central.hicache_init(num_steps=12, interval=1, first_enhance=0,
                                 backend="dmd", history=5)
    for step in range(5):
        state["step"] = step
        state["activated_steps"].append(step)
        value = _tree(float(step))
        central.hicache_update_tree(state, value)
        central.dmd_update_snapshots_tree(state, value)

    calls = 0
    original = central._dmd_fit_flat

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(central, "_dmd_fit_flat", counted)
    state["step"] = 6
    central.dmd_forecast_tree(state)
    central.dmd_forecast_tree(state)
    assert calls == 1

    state["step"] = 7
    central.dmd_forecast_tree(state)
    assert calls == 1

    state["step"] = 5
    state["activated_steps"].append(5)
    value = _tree(5.0)
    central.hicache_update_tree(state, value)
    central.dmd_update_snapshots_tree(state, value)
    state["step"] = 6
    central.dmd_forecast_tree(state)
    assert calls == 2


def test_dmd_uses_hermite_fallback_for_non_uniform_history():
    state = central.hicache_init(num_steps=10, interval=1, first_enhance=0,
                                 backend="dmd", history=5)
    for step in (0, 1, 3, 4):
        state["step"] = step
        state["activated_steps"].append(step)
        value = _tree(float(step))
        central.hicache_update_tree(state, value)
        central.dmd_update_snapshots_tree(state, value)

    state["step"] = 5
    forecast = central.dmd_forecast_tree(state)
    assert forecast["x"].shape == torch.Size([2])
    assert state["telemetry"]["fallbacks"]["dmd_nonuniform_or_short_tail"] == 1


def test_uncached_euler_identity_is_preserved():
    def dynamics(x, _t):
        return {"x": 0.25 * x["x"] + 1.0, "nested": {"y": x["nested"]["y"] - 0.5}}

    times = torch.linspace(0, 1, 7)
    initial = _tree(0.0)
    dense = list(Euler().solve_iter(dynamics, initial, times))[-1][0]
    no_op_cache = list(Euler().enable_dmd(interval=1).solve_iter(dynamics, initial, times))[-1][0]
    assert torch.equal(dense["x"], no_op_cache["x"])
    assert torch.equal(dense["nested"]["y"], no_op_cache["nested"]["y"])


def test_euler_records_actual_decisions_and_keeps_ss_solver_boundary():
    calls = 0

    def dynamics(x, t):
        nonlocal calls
        calls += 1
        return {"x": torch.ones_like(x["x"]), "nested": {"y": torch.ones_like(x["nested"]["y"])}}

    solver = Euler().enable_dmd(interval=3, first_enhance=0, history=5)
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 10)))
    telemetry = solver.get_hicache_telemetry()

    assert calls < 9
    assert telemetry["decisions"]["forecast"] > 0
    assert telemetry["decisions"]["full"] + telemetry["decisions"]["forecast"] == 9
    assert telemetry["method_counts"]

    calls = 0
    ss_solver = RungeKutta4().enable_dmd(interval=3, first_enhance=0)
    list(ss_solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 4)))
    assert calls == 12  # 3 steps x 4 RK dynamics evaluations; no HiCache on SS/native solvers.
    assert ss_solver.get_hicache_telemetry() is None


def test_benchmark_reports_missing_external_metrics():
    path = Path(__file__).parents[1] / "ab_accel_bench.py"
    spec = importlib.util.spec_from_file_location("fastsam3d_ab_bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    with pytest.raises(RuntimeError, match="FASTSAM3D_METRICS_PATH"):
        bench._load_metrics()
