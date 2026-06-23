"""
Simulation entry point (run on Mac/PC with MuJoCo installed).

Flow:
  1. Load MuJoCo model + open viewer
  2. Wait for ControlCommand from Jetson (or local control process)
  3. Step physics, reply with SimObservation
  4. Repeat
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from mujoco_env import RobotEnv
from comm.zmq_server import SimServer
from shared.protocol import SimObservation


def main():
    env = RobotEnv(render=True)
    server = SimServer()

    obs_dict = env.reset()
    env.start_viewer()
    print("[Sim] Ready. Waiting for control commands...")

    try:
        while True:
            cmd = server.recv_command(timeout_ms=50)
            if cmd is not None:
                obs_dict = env.step(cmd.joint_positions)
                obs = SimObservation(
                    joint_positions=obs_dict["joint_positions"],
                    joint_velocities=obs_dict["joint_velocities"],
                    timestamp=obs_dict["timestamp"],
                )
                server.send_observation(obs)
            else:
                # No command — keep physics running with zero control
                obs_dict = env.step([0.0] * env.n_joints)
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("[Sim] Shutting down.")
    finally:
        env.close()
        server.close()


if __name__ == "__main__":
    main()
