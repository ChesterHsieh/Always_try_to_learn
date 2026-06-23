"""
Control entry point (run on Jetson Orin Nano, or locally for testing).

Usage:
  python main.py --sim-host <IP of simulation PC>
  python main.py --sim-host localhost   # both sides on same machine
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from comm.zmq_client import ControlClient
from controller import PDController
from shared.protocol import ControlCommand


CONTROL_HZ = 50  # target control loop rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-host", default="localhost", help="IP of simulation PC")
    args = parser.parse_args()

    client = ControlClient(sim_host=args.sim_host)
    controller = PDController(n_joints=2)

    dt = 1.0 / CONTROL_HZ
    step = 0

    print(f"[Control] Running at {CONTROL_HZ} Hz → {args.sim_host}")
    try:
        while True:
            loop_start = time.time()

            # Build command (first step sends zeros to get initial obs)
            targets = controller.compute(obs=None) if step > 0 else [0.0, 0.0]
            cmd = ControlCommand(joint_positions=targets, timestamp=loop_start)

            obs = client.send_command(cmd)

            if step % CONTROL_HZ == 0:
                print(
                    f"[{step:5d}] targets={[f'{v:.3f}' for v in targets]} "
                    f"pos={[f'{v:.3f}' for v in obs.joint_positions]}"
                )

            step += 1
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("[Control] Shutting down.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
