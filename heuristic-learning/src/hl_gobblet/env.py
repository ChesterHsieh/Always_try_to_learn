"""GobbletEnv — gymnasium-style env from the P0 (controlled) player's view.

Spec: hl-gobblet-env, requirement "環境介面與隨機對手".

Design D4: single controlled player (P0). reset(seed) returns the opening
observation; step(action) applies P0's move and, if the game is not over, lets
an injected opponent object play P1, then returns the updated observation. Reward
is sparse from P0's perspective (+1 win / -1 loss / 0 draw or ongoing). info
carries a legal-action boolean mask over the fixed action space and an immutable
GobbletState snapshot for tracing / rendering.

Observation is a "god view" (full information, friendly to rule policies): per
cell, two owner channels per size slot, plus both reserves and a side-to-move
flag. A hidden-information variant is left for a later change.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from .moves import Move, action_space_size, index_to_move, move_to_index
from .rules import DEFAULT_MAX_MOVES, apply_move, legal_moves, status_of
from .state import BOARD_CELLS, GobbletState, Player, Size, initial_state

# obs layout: 9 cells * 3 sizes * 2 owner channels (54) + 2 reserves * 3 (6) + 1 flag
_OWNER_CHANNELS = 2
OBS_SIZE = BOARD_CELLS * len(Size) * _OWNER_CHANNELS + 2 * len(Size) + 1


class Opponent(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, state: GobbletState) -> Move: ...


def encode_observation(state: GobbletState) -> np.ndarray:
    """Flatten a GobbletState into the god-view float32 observation vector."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    k = 0
    for cell in state.board:
        for size in Size:
            owner = cell[size]
            if owner is Player.P0:
                obs[k] = 1.0
            elif owner is Player.P1:
                obs[k + 1] = 1.0
            k += _OWNER_CHANNELS
    for player in (Player.P0, Player.P1):
        for size in Size:
            obs[k] = float(state.reserve[player][size])
            k += 1
    obs[k] = float(state.current.value)
    return obs


def legal_action_mask(state: GobbletState) -> np.ndarray:
    """Boolean mask over the fixed action space; True where the move is legal."""
    mask = np.zeros(action_space_size(), dtype=bool)
    for move in legal_moves(state):
        mask[move_to_index(move)] = True
    return mask


class GobbletEnv:
    """Two-player Gobblet Gobblers as a single-agent env (P0 controlled)."""

    def __init__(
        self,
        opponent: Optional[Opponent] = None,
        *,
        reveal_loses: bool = False,
        max_moves: int = DEFAULT_MAX_MOVES,
    ) -> None:
        self.opponent = opponent
        self.reveal_loses = reveal_loses
        self.max_moves = max_moves
        self._state: GobbletState = initial_state()
        self._last_move: Optional[Move] = None

    @property
    def state(self) -> GobbletState:
        return self._state

    def reset(self, seed: int = 0) -> np.ndarray:
        self._state = initial_state(seed)
        self._last_move = None
        if self.opponent is not None:
            self.opponent.reset(seed)
        return encode_observation(self._state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply P0's action, let the opponent reply, return the SARS tuple.

        action is an index into the fixed action space; it must be legal for the
        current (P0) position.
        """
        move = index_to_move(action)
        self._state = apply_move(self._state, move, reveal_loses=self.reveal_loses)
        self._last_move = move

        st = status_of(self._state, max_moves=self.max_moves)
        if st.done:
            return self._finish(st)

        # Opponent (P1) replies.
        if self.opponent is not None:
            opp_move = self.opponent.act(self._state)
            self._state = apply_move(self._state, opp_move, reveal_loses=self.reveal_loses)
            self._last_move = opp_move
            st = status_of(self._state, max_moves=self.max_moves)
            if st.done:
                return self._finish(st)

        return encode_observation(self._state), 0.0, False, False, self._info()

    def _finish(self, st) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if st.draw or st.winner is None:
            reward = 0.0
        else:
            reward = 1.0 if st.winner is Player.P0 else -1.0
        terminated = not st.draw
        truncated = st.draw
        return encode_observation(self._state), reward, terminated, truncated, self._info()

    def info(self) -> dict[str, Any]:
        """Current legal mask, state snapshot, and last move (also used in step)."""
        return {
            "legal_mask": legal_action_mask(self._state),
            "state": self._state,
            "last_move": self._last_move,
        }

    # Backwards-compatible internal alias used by step().
    _info = info
