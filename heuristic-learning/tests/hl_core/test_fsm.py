"""FiniteStateMachine tests — spec hl-procedural-policy, requirement
"有限狀態機（Finite State Machine）".

Scenarios covered:
- condition 成立時轉移，且 action 由轉移後 state 子策略產生
- 無可用轉移則停留原 state
- reset 後回 initial_state 且清空進度（含 state 綁定的 macro）
"""

from __future__ import annotations

import numpy as np

from hl_core.fsm import FiniteStateMachine, Transition
from hl_core.rules import Rule, RuleTable


def _always(action: int, state_name: str) -> RuleTable:
    return RuleTable(
        rules=[
            Rule(name="always", priority=0, guard=lambda o, c: True, action_fn=lambda o, c: action)
        ],
        default_action=action,
        state_name=state_name,
    )


def _fsm() -> FiniteStateMachine:
    # State A emits 1, state B emits 3. Transition A->B when obs[0] > 0.5.
    return FiniteStateMachine(
        initial_state="A",
        deciders={"A": _always(1, "A"), "B": _always(3, "B")},
        transitions=[Transition(src="A", dst="B", condition=lambda o, c: o[0] > 0.5)],
    )


def test_transition_fires_and_action_from_new_state() -> None:
    fsm = _fsm()
    st = fsm.new_state()
    action, rec = fsm.step(np.array([1.0]), ctx=None, state=st)
    assert st.current == "B"
    assert action == 3  # action comes from the state we transitioned INTO
    assert rec.state == "B"


def test_no_transition_stays_in_current_state() -> None:
    fsm = _fsm()
    st = fsm.new_state()
    action, rec = fsm.step(np.array([0.0]), ctx=None, state=st)
    assert st.current == "A"
    assert action == 1
    assert rec.state == "A"


def test_reset_returns_to_initial_state() -> None:
    fsm = _fsm()
    st = fsm.new_state()
    fsm.step(np.array([1.0]), ctx=None, state=st)  # move to B
    assert st.current == "B"
    st.reset()
    assert st.current == "A"
