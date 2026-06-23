"""
Shared message protocol between control (Jetson) and simulation (MuJoCo).
Both sides import this to stay in sync on schema.
"""

from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class ControlCommand:
    """Control → Simulation: joint position targets."""
    joint_positions: List[float]
    timestamp: float

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControlCommand":
        d = json.loads(data)
        return cls(**d)


@dataclass
class SimObservation:
    """Simulation → Control: current robot state."""
    joint_positions: List[float]
    joint_velocities: List[float]
    timestamp: float
    done: bool = False

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "SimObservation":
        d = json.loads(data)
        return cls(**d)


# ZMQ addresses — control connects to sim as server
SIM_CMD_ADDR = "tcp://{}:5555"   # sim listens for commands
SIM_OBS_ADDR = "tcp://{}:5556"   # sim publishes observations
