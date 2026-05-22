"""Trace invariants for hl-procedural-policy.

Task 2.2 covers TraceRecord (frozen, full fields). Task 6.1 adds the
ProceduralPolicy-level invariants: HeuristicPolicy subtyping, determinism,
no-side-effect export, immutable snapshot.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hl_core.fsm import FiniteStateMachine, Transition
from hl_core.policy import ProceduralPolicy
from hl_core.rules import Rule, RuleTable
from hl_core.trace import TraceRecord
from hl_lander.policy import HeuristicPolicy


def test_trace_record_is_frozen() -> None:
    rec = TraceRecord(step=0, state="descend", rule="r1", macro_active=False, action=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.action = 3  # type: ignore[misc]


def test_trace_record_has_all_semantic_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(TraceRecord)}
    assert field_names == {"step", "state", "rule", "macro_active", "action"}


def _toy_policy() -> ProceduralPolicy:
    # A>B on obs[0]>0.5; A emits 1, B emits 3. Trace records step/state/rule.
    a = RuleTable(
        rules=[Rule(name="a", priority=0, guard=lambda o, c: True, action_fn=lambda o, c: 1)],
        default_action=1,
        state_name="A",
    )
    b = RuleTable(
        rules=[Rule(name="b", priority=0, guard=lambda o, c: True, action_fn=lambda o, c: 3)],
        default_action=3,
        state_name="B",
    )
    fsm = FiniteStateMachine(
        initial_state="A",
        deciders={"A": a, "B": b},
        transitions=[Transition(src="A", dst="B", condition=lambda o, c: o[0] > 0.5)],
    )
    return ProceduralPolicy(fsm=fsm)


def _run(policy: ProceduralPolicy, observations: list[np.ndarray], seed: int = 0) -> list[int]:
    policy.reset(seed)
    return [policy.act(o) for o in observations]


def test_procedural_policy_is_heuristic_policy() -> None:
    assert isinstance(_toy_policy(), HeuristicPolicy)


def test_decision_trace_length_equals_steps() -> None:
    policy = _toy_policy()
    obs = [np.array([0.0]), np.array([1.0]), np.array([1.0])]
    _run(policy, obs)
    assert len(policy.decision_trace()) == 3


def test_same_seed_reproduces_identical_trace() -> None:
    obs = [np.array([0.0]), np.array([1.0]), np.array([0.0])]
    p1, p2 = _toy_policy(), _toy_policy()
    _run(p1, obs, seed=7)
    _run(p2, obs, seed=7)
    assert p1.decision_trace() == p2.decision_trace()


def test_trace_export_has_no_side_effect() -> None:
    obs = [np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])]
    no_export = _toy_policy()
    with_export = _toy_policy()

    no_export.reset(3)
    with_export.reset(3)
    out_a, out_b = [], []
    for o in obs:
        out_a.append(no_export.act(o))
        with_export.decision_trace()  # mid-episode export must not change behaviour
        out_b.append(with_export.act(o))
    assert out_a == out_b


def test_decision_trace_returns_immutable_snapshot() -> None:
    policy = _toy_policy()
    _run(policy, [np.array([0.0])])
    trace = policy.decision_trace()
    assert isinstance(trace, tuple)  # snapshot, not the internal mutable buffer
