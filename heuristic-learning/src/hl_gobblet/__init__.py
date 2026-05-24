"""hl_gobblet — 奇迹连连 (Gobblet Gobblers) environment for the HL paradigm.

Capability: hl-gobblet-env. A discrete, turn-based, two-player board game with
hidden information (larger pieces gobble — cover — smaller ones). Built as a
gradient-free, code-as-policy environment: immutable state, pure-function
transitions, deterministic legal-move generation.

HL red line: no gradients, no neural-network weights. Depends only on stdlib +
numpy (+ rich for the CLI viewer) and MUST NOT import hl_lander or any other
concrete environment.
"""

from __future__ import annotations

from .moves import Move, MoveKind, action_space_size, index_to_move, move_to_index
from .rules import DEFAULT_MAX_MOVES, Status, apply_move, legal_moves, line_winner, status_of
from .state import GobbletState, Player, Size, initial_state, top_owner

__all__ = [
    "Player",
    "Size",
    "GobbletState",
    "initial_state",
    "top_owner",
    "Move",
    "MoveKind",
    "move_to_index",
    "index_to_move",
    "action_space_size",
    "legal_moves",
    "apply_move",
    "line_winner",
    "status_of",
    "Status",
    "DEFAULT_MAX_MOVES",
]
