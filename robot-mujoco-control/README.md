# robot-mujoco-control

Two-sided robot control framework:

| Side | Runs on | Role |
|------|---------|------|
| `simulation/` | Mac / PC | MuJoCo physics + viewer |
| `control/` | Jetson Orin Nano | Controller policy → sends joint targets |

Communication: **ZeroMQ REQ/REP** over TCP (port 5555)

```
Jetson (control)          Mac/PC (simulation)
 ControlCommand  ──────►  RobotEnv.step()
 SimObservation  ◄──────  MuJoCo physics
```

---

## Phase 1: Local smoke test (both sides on your Mac)

```bash
# Terminal 1 — start simulation
cd simulation
pip install -r requirements.txt
python src/main.py

# Terminal 2 — start controller
cd control
pip install -r requirements.txt
python src/main.py --sim-host localhost
```

You should see the 2-DOF arm swinging in the MuJoCo viewer.

---

## Phase 2: Jetson Orin Nano → Mac

1. Both devices must be on the same LAN (or SSH tunnel)
2. Find Mac's IP: `ipconfig getifaddr en0`
3. On Mac: `python simulation/src/main.py`
4. On Jetson:
   ```bash
   python control/src/main.py --sim-host <MAC_IP>
   ```

---

## Project structure

```
robot-mujoco-control/
├── shared/
│   └── protocol.py          # ControlCommand / SimObservation schemas
├── simulation/
│   ├── models/
│   │   └── simple_arm.xml   # 2-DOF MJCF model
│   ├── src/
│   │   ├── main.py          # Sim entry point
│   │   ├── mujoco_env.py    # MuJoCo wrapper
│   │   └── comm/
│   │       └── zmq_server.py
│   └── requirements.txt
└── control/
    ├── src/
    │   ├── main.py          # Control entry point
    │   ├── controller.py    # PD / policy (replace with your RL policy)
    │   └── comm/
    │       └── zmq_client.py
    └── requirements.txt
```

---

## Roadmap

- [x] Phase 1 — Local MuJoCo simulation controlled via ZMQ
- [ ] Phase 2 — Jetson ↔ Mac over LAN
- [ ] Phase 3 — Replace PD controller with learned policy (RL / imitation)
- [ ] Phase 4 — Real robot hardware in the loop
