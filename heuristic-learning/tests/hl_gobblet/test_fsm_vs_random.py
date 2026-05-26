"""Win-rate regression: FsmGobbletV1 must beat RandomOpponent decisively.

Spec: hl-gobblet-fsm-controller, requirement "棋力門檻與 Golden Trace 回歸"
(scenario 勝過隨機對手). Plays a fixed set of seeds with the FSM on both sides so
the result is not an artefact of moving first, and asserts the FSM win rate is
well over half. The threshold is set with headroom below the measured 100% so
benign future tweaks do not flake, while a real regression to ~random strength
still trips it.
"""

from __future__ import annotations

from hl_gobblet.controllers import FsmGobbletV1
from hl_gobblet.opponents import RandomOpponent
from hl_gobblet.rules import DEFAULT_MAX_MOVES, apply_move, status_of
from hl_gobblet.state import Player, initial_state

_N_SEEDS = 50
_WIN_RATE_FLOOR = 0.80  # measured 100%; floor leaves headroom but stays decisive


def _play(seed: int, fsm_side: Player):
    fsm = FsmGobbletV1()
    fsm.reset(seed)
    rnd = RandomOpponent(seed=seed + 1000)
    rnd.reset(seed + 1000)
    s = initial_state(seed)
    p0 = fsm if fsm_side is Player.P0 else rnd
    p1 = fsm if fsm_side is Player.P1 else rnd
    for _ in range(DEFAULT_MAX_MOVES + 1):
        st = status_of(s)
        if st.done:
            return st
        mover = p0 if s.current is Player.P0 else p1
        s = apply_move(s, mover.act(s))
    return status_of(s)


def test_fsm_beats_random_decisively():
    wins = 0
    games = 0
    for seed in range(_N_SEEDS):
        for side in (Player.P0, Player.P1):
            st = _play(seed, side)
            games += 1
            if not st.draw and st.winner is side:
                wins += 1
    rate = wins / games
    assert rate >= _WIN_RATE_FLOOR, f"fsm win rate {rate:.0%} below floor {_WIN_RATE_FLOOR:.0%}"
