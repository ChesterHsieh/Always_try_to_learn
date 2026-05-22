"""MacroAction tests — spec hl-procedural-policy, requirement "巨集動作（Macro-Action）".

Scenarios covered:
- 定長 macro 連續 N 步依序輸出，第 N 步後 is_active() 為 False
- 執行中 interrupt 為真時於第 k 步立即終止並交還控制權
- reset 後進度歸零

Design D1: the MacroAction config is frozen; per-episode progress lives in a
separate MacroState the policy owns and resets.
"""

from __future__ import annotations

import numpy as np

from hl_core.macros import MacroAction, MacroState


def test_fixed_length_macro_emits_sequence_then_inactive() -> None:
    macro = MacroAction(name="burn", sequence=(2, 2, 0))
    st = MacroState()
    macro.start(st)
    emitted = []
    while macro.is_active(st):
        emitted.append(macro.next_action(np.zeros(8), ctx=None, state=st))
    assert emitted == [2, 2, 0]
    assert macro.is_active(st) is False


def test_interrupt_terminates_midway_and_yields_control() -> None:
    # interrupt fires once obs[6] (leg contact) becomes truthy.
    macro = MacroAction(
        name="burn",
        sequence=(2, 2, 2, 2),
        interrupt=lambda o, c: bool(o[6]),
    )
    st = MacroState()
    macro.start(st)

    no_contact = np.zeros(8)
    contact = np.zeros(8)
    contact[6] = 1.0

    a0 = macro.next_action(no_contact, ctx=None, state=st)  # step 0, no interrupt
    assert a0 == 2 and macro.is_active(st)

    # Before step 1 the interrupt condition holds -> macro stops, control returns.
    assert macro.should_interrupt(contact, ctx=None) is True
    macro.stop(st)
    assert macro.is_active(st) is False


def test_reset_clears_progress() -> None:
    macro = MacroAction(name="burn", sequence=(2, 2))
    st = MacroState()
    macro.start(st)
    macro.next_action(np.zeros(8), ctx=None, state=st)
    st.reset()
    assert macro.is_active(st) is False
