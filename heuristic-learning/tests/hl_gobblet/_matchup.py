"""Deterministic head-to-head match harness for Gobblet controllers.

Capability: hl-gobblet-fsm-controller-v2, requirements "棋力門檻——大機率擊敗 v1"
and "3×3 交互對打結果矩陣". Pure functions that play controllers against each
other and compute win rates. Everything is deterministic: a given (a, b, seed,
side) replays the exact same game, so a win-rate floor is a stable regression.

Controllers are passed as zero-arg factories so each game gets a fresh instance.
RandomOpponent is seeded off the game seed so its play varies per seed but is
reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from hl_gobblet.controllers import FsmGobbletV1, FsmGobbletV2
from hl_gobblet.opponents import RandomOpponent
from hl_gobblet.rules import DEFAULT_MAX_MOVES, Status, apply_move, legal_moves, status_of
from hl_gobblet.state import GobbletState, Player, initial_state

# A factory takes the game seed and returns a fresh opponent for one game.
Factory = Callable[[int], object]

# Both FSM controllers are deterministic and the opening position is
# seed-independent, so two deterministic controllers would replay the SAME game
# for every seed (only 2 distinct games over all seeds: each as P0 / P1). To turn
# "many seeds" into a genuine distribution of distinct positions, we play a few
# seeded *random* legal plies first, then hand the (varied) position to the two
# controllers. This keeps the controllers themselves deterministic while sampling
# the position space. Named constant per the HL red line.
_DEFAULT_RANDOM_OPENING_PLIES = 4


def random_factory(seed: int):
    """A RandomOpponent seeded off the game seed (reproducible, varies per seed)."""
    opp = RandomOpponent(seed=seed + 1000)
    opp.reset(seed + 1000)
    return opp


def v1_factory(seed: int):
    opp = FsmGobbletV1()
    opp.reset(seed)
    return opp


def v2_factory(seed: int):
    opp = FsmGobbletV2()
    opp.reset(seed)
    return opp


# Registry used by the 3x3 matrix (stable, named order).
FACTORIES: dict[str, Factory] = {
    "random": random_factory,
    "v1": v1_factory,
    "v2": v2_factory,
}


def _random_opening(seed: int, plies: int) -> GobbletState | None:
    """Play `plies` seeded random legal moves from the opening; return the
    resulting position, or None if the game already ended during the opening
    (such a seed is skipped so both controllers get a live position to play)."""
    rng = np.random.default_rng(seed)
    s = initial_state(seed)
    for _ in range(plies):
        if status_of(s).done:
            return None
        moves = legal_moves(s)
        s = apply_move(s, moves[int(rng.integers(len(moves)))])
    return None if status_of(s).done else s


def play_match(
    p0_factory: Factory,
    p1_factory: Factory,
    seed: int,
    *,
    opening_plies: int = _DEFAULT_RANDOM_OPENING_PLIES,
) -> Status | None:
    """Play one full game: p0_factory drives P0, p1_factory drives P1.

    A seeded random opening of `opening_plies` legal moves is played first so each
    seed yields a distinct starting position. Returns the terminal Status, or None
    if the seed's random opening already ended the game (skip it). Deterministic
    for a given (p0_factory, p1_factory, seed, opening_plies).
    """
    start = _random_opening(seed, opening_plies)
    if start is None:
        return None
    p0 = p0_factory(seed)
    p1 = p1_factory(seed)
    s = start
    for _ in range(DEFAULT_MAX_MOVES + 1):
        st = status_of(s)
        if st.done:
            return st
        mover = p0 if s.current is Player.P0 else p1
        s = apply_move(s, mover.act(s))
    return status_of(s)


def winrate(
    a: Factory, b: Factory, seeds, *, opening_plies: int = _DEFAULT_RANDOM_OPENING_PLIES
) -> float:
    """Win rate of controller `a` against `b` over `seeds`, alternating sides.

    For each seed, `a` plays once as P0 (vs `b` as P1) and once as P1 from the
    seed's random opening, so the result is not an artefact of moving first. Seeds
    whose opening already decided the game are skipped. Draws count as non-wins.
    Returns a wins / total played games in [0, 1].
    """
    a_wins = 0
    games = 0
    for seed in seeds:
        st = play_match(a, b, seed, opening_plies=opening_plies)
        if st is not None:
            games += 1
            if not st.draw and st.winner is Player.P0:
                a_wins += 1
        st = play_match(b, a, seed, opening_plies=opening_plies)
        if st is not None:
            games += 1
            if not st.draw and st.winner is Player.P1:
                a_wins += 1
    return a_wins / games if games else 0.0
