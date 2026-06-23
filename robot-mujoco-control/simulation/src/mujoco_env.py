import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / "models" / "simple_arm.xml"


class RobotEnv:
    """Thin MuJoCo wrapper: load model, step physics, read state."""

    def __init__(self, model_path: str = str(MODEL_PATH), render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._viewer = None
        self.n_joints = self.model.nu  # number of actuators

    def start_viewer(self):
        if self._render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, joint_positions: list[float]) -> dict:
        """Apply position targets and advance one physics step."""
        self.data.ctrl[:] = np.array(joint_positions[: self.n_joints])
        mujoco.mj_step(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
        return self._get_obs()

    def _get_obs(self) -> dict:
        return {
            "joint_positions": self.data.qpos[: self.n_joints].tolist(),
            "joint_velocities": self.data.qvel[: self.n_joints].tolist(),
            "timestamp": self.data.time,
        }

    def close(self):
        if self._viewer:
            self._viewer.close()
