"""HeuristicPolicy — the one stable contract every HL controller implements.

Spec: openspec/specs (capability hl-lunar-lander), requirement "HeuristicPolicy 抽象介面".

Deliberately narrow: only `reset(seed)` and `act(observation)`. There is NO
`update`/`learn`/`train` method — in the HL paradigm "learning" happens by
editing controller source code (a new baseline_v{n}.py file), not by an
in-process update call.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HeuristicPolicy(Protocol):
    def reset(self, seed: int) -> None:
        """Re-initialize all internal state so a given seed reproduces a trajectory.

        Stateless controllers may make this a no-op. Stateful ones (future CPG
        phase, MPC buffer, CV state machine) MUST reset every field here.
        """
        ...

    def act(self, observation: np.ndarray) -> int:
        """Map a LunarLander-v3 observation (8-dim) to a discrete action in {0,1,2,3}."""
        ...
