"""MacroAction — a fixed-length (optionally interruptible) action sequence.

Spec: hl-procedural-policy, requirement "巨集動作（Macro-Action）".

Design D1: the MacroAction is a frozen config (its action sequence + optional
interrupt predicate). Per-episode execution progress lives in a separate mutable
MacroState that the ProceduralPolicy owns and clears in reset(). One MacroState
tracks one macro; an FSM with several macro states gives each its own MacroState.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

Interrupt = Callable[[Any, Any], bool]


class MacroState:
    """Mutable per-episode progress for a single MacroAction.

    Kept out of the frozen MacroAction so reset() can zero it in one place and a
    given seed reproduces the same trajectory.
    """

    def __init__(self) -> None:
        self._index = 0
        self._active = False

    def reset(self) -> None:
        self._index = 0
        self._active = False


@dataclass(frozen=True)
class MacroAction:
    """An immutable macro: a discrete action sequence run as one decision unit."""

    name: str
    sequence: tuple[int, ...]
    interrupt: Optional[Interrupt] = None

    def start(self, state: MacroState) -> None:
        state._index = 0
        state._active = len(self.sequence) > 0

    def is_active(self, state: MacroState) -> bool:
        return state._active and state._index < len(self.sequence)

    def should_interrupt(self, observation: Any, ctx: Any) -> bool:
        return self.interrupt is not None and bool(self.interrupt(observation, ctx))

    def stop(self, state: MacroState) -> None:
        state._active = False

    def next_action(self, observation: Any, ctx: Any, state: MacroState) -> int:
        """Emit the next action in the sequence and advance progress.

        Callers check should_interrupt() / is_active() before calling; this only
        advances the cursor and deactivates once the sequence is exhausted.
        """
        action = int(self.sequence[state._index])
        state._index += 1
        if state._index >= len(self.sequence):
            state._active = False
        return action
