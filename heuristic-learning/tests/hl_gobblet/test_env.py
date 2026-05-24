"""Tests for GobbletEnv (spec: 環境介面與隨機對手)."""

from __future__ import annotations

import numpy as np

from hl_gobblet.env import GobbletEnv, encode_observation
from hl_gobblet.moves import move_to_index
from hl_gobblet.opponents import RandomOpponent
from hl_gobblet.rules import legal_moves
from hl_gobblet.state import initial_state


def _p0_pick(env: GobbletEnv, rng: np.random.Generator) -> int:
    """Pick a legal P0 action index from the env's current state."""
    moves = legal_moves(env.state)
    move = moves[int(rng.integers(len(moves)))]
    return move_to_index(move)


def test_reset_exposes_legal_actions():
    """Scenario: reset 後可取得合法動作."""
    env = GobbletEnv(opponent=RandomOpponent(seed=0))
    obs = env.reset(seed=0)
    assert obs.shape[0] > 0
    np.testing.assert_array_equal(obs, encode_observation(initial_state()))
    mask = env.info()["legal_mask"]
    assert mask.any()


def test_step_applies_p0_then_opponent_replies():
    """Scenario: step 推進並讓對手回應."""
    env = GobbletEnv(opponent=RandomOpponent(seed=3))
    env.reset(seed=3)
    # P0 places small on cell 0.
    from hl_gobblet.moves import Move
    from hl_gobblet.state import Player, Size, top_owner

    action = move_to_index(Move.place(Size.SMALL, 0))
    _obs, _reward, terminated, truncated, info = env.step(action)
    state = info["state"]
    assert top_owner(state.board[0]) is Player.P0  # P0's move applied
    if not (terminated or truncated):
        # After a non-terminal P0 move the opponent (P1) has replied, so it's
        # P0's turn again.
        assert state.current is Player.P0


def test_same_seed_same_action_sequence_is_deterministic():
    """Scenario: 相同seed決定性重現."""

    def run() -> list:
        env = GobbletEnv(opponent=RandomOpponent(seed=11))
        env.reset(seed=11)
        rng = np.random.default_rng(123)  # drives P0 identically across runs
        trace = []
        for _ in range(60):
            action = _p0_pick(env, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            trace.append((tuple(obs.tolist()), reward, terminated, truncated))
            if terminated or truncated:
                break
        return trace  # noqa: RET504 -- explicit for readability

    assert run() == run()


def test_random_self_play_terminates_without_illegal_moves():
    """Scenario: 隨機對局可正常終止 — two RandomOpponents play to the end."""
    env = GobbletEnv(opponent=RandomOpponent(seed=2))
    env.reset(seed=1)
    p0 = RandomOpponent(seed=1)
    terminated = truncated = False
    for _ in range(env.max_moves + 5):
        # P0 driven by its own RandomOpponent, picking from current legal moves.
        moves = legal_moves(env.state)
        assert moves  # never stuck with no legal move mid-game
        action = move_to_index(p0.act(env.state))
        _obs, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated  # game ended (win or draw cap)
