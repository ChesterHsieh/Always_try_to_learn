"""RuleTable tests — spec hl-procedural-policy, requirement "規則表（Rule Table）".

Scenarios covered:
- 多條規則命中時取最高優先序 (lowest priority number wins)
- 沒有任何 guard 命中 -> default_action, trace rule == "default"
- guard/action 為純函式：連兩次同 obs 同 action 且不就地修改 obs
"""

from __future__ import annotations

import numpy as np

from hl_core.rules import Rule, RuleTable


def _table() -> RuleTable:
    # Two rules whose guards both hold for obs[0] > 0; priority 10 must win over 20.
    return RuleTable(
        rules=[
            Rule(name="hi", priority=20, guard=lambda o, c: o[0] > 0, action_fn=lambda o, c: 3),
            Rule(name="lo", priority=10, guard=lambda o, c: o[0] > 0, action_fn=lambda o, c: 1),
        ],
        default_action=0,
    )


def test_highest_priority_rule_wins() -> None:
    action, rec = _table().decide(np.array([1.0]), ctx=None)
    assert action == 1
    assert rec.rule == "lo"


def test_no_guard_matches_falls_back_to_default() -> None:
    action, rec = _table().decide(np.array([-1.0]), ctx=None)
    assert action == 0
    assert rec.rule == "default"


def test_decide_is_pure_no_mutation_deterministic() -> None:
    obs = np.array([1.0, 2.0, 3.0])
    before = obs.copy()
    a1, _ = _table().decide(obs, ctx=None)
    a2, _ = _table().decide(obs, ctx=None)
    assert a1 == a2
    assert np.array_equal(obs, before)  # decide must not mutate the observation
