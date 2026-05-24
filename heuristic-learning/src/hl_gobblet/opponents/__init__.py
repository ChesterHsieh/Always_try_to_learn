"""Opponent objects for GobbletEnv — pick a move from the current state.

An opponent exposes `reset(seed)` and `act(state) -> Move`. This package ships
only RandomOpponent (used to validate that the env can self-play a full game);
hand-written rule controllers and greedy look-ahead are left for a later change.
"""

from __future__ import annotations

from .random import RandomOpponent

__all__ = ["RandomOpponent"]
