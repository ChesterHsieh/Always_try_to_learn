"""RuleTable — a priority-ordered list of (guard, action) pure-function rules.

Spec: hl-procedural-policy, requirement "規則表（Rule Table）".

decide() returns the action of the first rule (lowest priority number) whose
guard holds, plus a TraceRecord naming the fired rule. With no match it returns
default_action with rule="default" rather than raising. guard and action_fn MUST
be side-effect-free pure functions (Design D1) — the table holds no execution
state, so it is safe to share across policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .trace import TraceRecord

Guard = Callable[[Any, Any], bool]
ActionFn = Callable[[Any, Any], int]

# Sentinel state name for a rule table used as the policy root (no enclosing FSM).
ROOT_STATE = "<root>"


@dataclass(frozen=True)
class Rule:
    """One priority-ordered rule. Lower `priority` number = higher precedence."""

    name: str
    priority: int
    guard: Guard
    action_fn: ActionFn


@dataclass(frozen=True)
class RuleTable:
    """Immutable rule set. `state_name` labels the TraceRecord (set by an owning FSM)."""

    rules: list[Rule]
    default_action: int
    state_name: str = ROOT_STATE
    _sorted: tuple[Rule, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Pre-sort once; frozen dataclass requires object.__setattr__.
        ordered = tuple(sorted(self.rules, key=lambda r: r.priority))
        object.__setattr__(self, "_sorted", ordered)

    def decide(self, observation: Any, ctx: Any) -> tuple[int, TraceRecord]:
        for rule in self._sorted:
            if rule.guard(observation, ctx):
                action = int(rule.action_fn(observation, ctx))
                return action, self._record(ctx, rule.name, action)
        action = int(self.default_action)
        return action, self._record(ctx, "default", action)

    def _record(self, ctx: Any, rule_name: str, action: int) -> TraceRecord:
        step = getattr(ctx, "step_index", 0)
        return TraceRecord(
            step=step,
            state=self.state_name,
            rule=rule_name,
            macro_active=False,
            action=action,
        )
