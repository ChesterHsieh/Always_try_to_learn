"""Tests for RandomOpponent (spec: 環境介面與隨機對手, opponent part)."""

from __future__ import annotations

from hl_gobblet.opponents import RandomOpponent
from hl_gobblet.rules import apply_move, legal_moves, status_of
from hl_gobblet.state import initial_state


def test_same_seed_same_choice():
    s = initial_state()
    a = RandomOpponent(seed=42)
    b = RandomOpponent(seed=42)
    # Same seed, same state -> identical sequence of picks.
    for _ in range(5):
        assert a.act(s) == b.act(s)


def test_reset_reproduces_sequence():
    s = initial_state()
    opp = RandomOpponent(seed=7)
    first = [opp.act(s) for _ in range(5)]
    opp.reset(7)
    again = [opp.act(s) for _ in range(5)]
    assert first == again


def test_never_returns_illegal_move():
    """Walk a full self-play game; every move chosen must be legal."""
    s = initial_state()
    p0 = RandomOpponent(seed=1)
    p1 = RandomOpponent(seed=2)
    for ply in range(60):
        st = status_of(s)
        if st.done:
            break
        mover = p0 if s.current.value == 0 else p1
        move = mover.act(s)
        assert move in legal_moves(s)
        s = apply_move(s, move)
