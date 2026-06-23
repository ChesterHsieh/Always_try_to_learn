"""
ZMQ client running on Jetson Orin Nano.
Sends ControlCommand to sim, receives SimObservation in reply.
"""

import zmq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.protocol import ControlCommand, SimObservation, SIM_CMD_ADDR


class ControlClient:
    def __init__(self, sim_host: str):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        addr = SIM_CMD_ADDR.format(sim_host)
        self._sock.connect(addr)
        print(f"[ControlClient] Connected to {addr}")

    def send_command(self, cmd: ControlCommand) -> SimObservation:
        self._sock.send(cmd.to_bytes())
        return SimObservation.from_bytes(self._sock.recv())

    def close(self):
        self._sock.close()
        self._ctx.term()
