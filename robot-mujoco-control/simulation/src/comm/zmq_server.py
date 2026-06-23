"""
ZMQ server running on the simulation (PC/Mac) side.
- REP socket on port 5555: receives ControlCommand, replies with SimObservation
"""

import zmq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.protocol import ControlCommand, SimObservation, SIM_CMD_ADDR


class SimServer:
    def __init__(self, host: str = "*"):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(SIM_CMD_ADDR.format(host))
        print(f"[SimServer] Listening on {SIM_CMD_ADDR.format(host)}")

    def recv_command(self, timeout_ms: int = 100) -> ControlCommand | None:
        """Non-blocking receive. Returns None if nothing arrived."""
        if self._sock.poll(timeout_ms):
            return ControlCommand.from_bytes(self._sock.recv())
        return None

    def send_observation(self, obs: SimObservation):
        self._sock.send(obs.to_bytes())

    def close(self):
        self._sock.close()
        self._ctx.term()
