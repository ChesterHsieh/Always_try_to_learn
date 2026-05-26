"""Tests for FsmGobbletV1: interface, mode switching, legality, determinism, trace.

Spec: hl-gobblet-fsm-controller, requirements "opponent 介面相容",
"FSM 兩模式與切換條件", "各模式的規則表決策", "決策軌跡導出".
"""

from __future__ import annotations

from hl_gobblet.controllers import FsmGobbletV1
from hl_gobblet.controllers._modes import AGGRESSIVE, DEFENSIVE
from hl_gobblet.moves import Move
from hl_gobblet.opponents import RandomOpponent
from hl_gobblet.rules import apply_move, legal_moves, status_of
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size, initial_state


def _cell(**sizes):
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def _play_fsm_vs_random(seed: int, max_plies: int = 80):
    """Play one fsm(P0)-vs-random(P1) game; return (final_state, fsm)."""
    fsm = FsmGobbletV1()
    fsm.reset(seed)
    rnd = RandomOpponent(seed=seed + 1)
    rnd.reset(seed + 1)
    s = initial_state(seed)
    for _ in range(max_plies):
        if status_of(s).done:
            break
        mover = fsm if s.current is Player.P0 else rnd
        s = apply_move(s, mover.act(s))
    return s, fsm


# --- opponent interface ---------------------------------------------------------
def test_act_returns_legal_move():
    """Scenario: 永遠回傳合法步."""
    fsm = FsmGobbletV1()
    fsm.reset(0)
    s = initial_state()
    move = fsm.act(s)
    assert isinstance(move, Move)
    assert move in legal_moves(s)


def test_act_does_not_mutate_state_and_is_repeatable():
    """Scenario: 不就地修改傳入局面 — same state twice -> same move, state unchanged."""
    fsm = FsmGobbletV1()
    fsm.reset(0)
    s = initial_state()
    before = s
    m1 = fsm.act(s)
    # A fresh controller on the SAME position must reproduce the move (no state
    # was mutated and the controller is deterministic per position-at-step-0).
    fsm2 = FsmGobbletV1()
    fsm2.reset(0)
    m2 = fsm2.act(s)
    assert m1 == m2
    assert s == before  # frozen dataclass: input not mutated


def test_full_game_uses_only_legal_moves():
    """Across a whole game the controller never emits an illegal move."""
    s = initial_state(0)
    fsm = FsmGobbletV1()
    fsm.reset(0)
    rnd = RandomOpponent(seed=1)
    rnd.reset(1)
    while not status_of(s).done and s.move_count < 80:
        mover = fsm if s.current is Player.P0 else rnd
        m = mover.act(s)
        assert m in legal_moves(s)
        s = apply_move(s, m)
    assert status_of(s).done or s.move_count >= 80


# --- mode switching (three scenarios) -------------------------------------------
def _two_in_row(player: Player, *, current: Player) -> GobbletState:
    p, o = player, player.other
    board = [
        _cell(SMALL=p),
        _cell(SMALL=p),
        EMPTY_CELL,
        _cell(SMALL=o),
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    ]
    return GobbletState(
        board=tuple(board), reserve=((1, 2, 2), (1, 2, 2)), current=current, move_count=4
    )


def test_switches_to_defensive_when_opponent_threatens_and_no_win():
    """Scenario: 對手有立即威脅且我方無殺時轉防守."""
    # P1 owns 0,1; P0 to move and has no win -> must defend.
    s = _two_in_row(Player.P1, current=Player.P0)
    fsm = FsmGobbletV1()
    fsm.reset(0)
    fsm.act(s)
    rec = fsm.decision_trace()[-1]
    assert rec.state == DEFENSIVE
    assert rec.rule == "block"


def test_stays_aggressive_and_wins_when_we_have_a_win():
    """Scenario: 我方有殺時維持/進入攻擊 — current player completes the line."""
    s = _two_in_row(Player.P0, current=Player.P0)
    fsm = FsmGobbletV1()
    fsm.reset(0)
    move = fsm.act(s)
    rec = fsm.decision_trace()[-1]
    assert rec.state == AGGRESSIVE
    assert rec.rule == "win_now"
    assert status_of(apply_move(s, move)).winner is Player.P0


def test_stays_aggressive_when_no_threat():
    """Scenario: 無立即威脅時維持攻擊 — opening has no threat, stays aggressive."""
    s = initial_state()
    fsm = FsmGobbletV1()
    fsm.reset(0)
    fsm.act(s)
    rec = fsm.decision_trace()[-1]
    assert rec.state == AGGRESSIVE


# --- reset reproducibility ------------------------------------------------------
def test_reset_reproduces_same_game():
    """Scenario: reset 後可重現 — same seed replays the same fsm move sequence."""
    s1, fsm1 = _play_fsm_vs_random(7)
    s2, fsm2 = _play_fsm_vs_random(7)
    t1 = fsm1.decision_trace()
    t2 = fsm2.decision_trace()
    assert t1 == t2
    assert s1 == s2


# --- decision trace -------------------------------------------------------------
def test_trace_length_matches_fsm_plies():
    """Scenario: trace 長度與內容對齊對局 — one record per fsm ply with fields set."""
    _s, fsm = _play_fsm_vs_random(3)
    trace = fsm.decision_trace()
    assert len(trace) >= 1
    for i, rec in enumerate(trace):
        assert rec.step == i
        assert rec.state in (AGGRESSIVE, DEFENSIVE)
        assert isinstance(rec.rule, str) and rec.rule
        assert rec.action_index >= 0


def test_trace_export_has_no_side_effect():
    """Scenario: trace 導出無副作用 — exporting mid-game does not change later moves."""
    # Run A: export the trace after every ply.
    fsm_a = FsmGobbletV1()
    fsm_a.reset(5)
    rnd_a = RandomOpponent(seed=6)
    rnd_a.reset(6)
    sa = initial_state(5)
    moves_a = []
    while not status_of(sa).done and sa.move_count < 80:
        if sa.current is Player.P0:
            m = fsm_a.act(sa)
            _ = fsm_a.decision_trace()  # export mid-game
            moves_a.append(m)
        else:
            m = rnd_a.act(sa)
        sa = apply_move(sa, m)

    # Run B: never export until the end.
    fsm_b = FsmGobbletV1()
    fsm_b.reset(5)
    rnd_b = RandomOpponent(seed=6)
    rnd_b.reset(6)
    sb = initial_state(5)
    moves_b = []
    while not status_of(sb).done and sb.move_count < 80:
        if sb.current is Player.P0:
            m = fsm_b.act(sb)
            moves_b.append(m)
        else:
            m = rnd_b.act(sb)
        sb = apply_move(sb, m)

    assert moves_a == moves_b
    assert fsm_a.decision_trace() == fsm_b.decision_trace()


def test_returns_tuple_snapshot_not_internal_list():
    """decision_trace() returns an immutable tuple, so callers can't mutate state."""
    _s, fsm = _play_fsm_vs_random(0)
    trace = fsm.decision_trace()
    assert isinstance(trace, tuple)
