"""hl_lander.rl — JAX/flax RL baseline (PPO) used ONLY as a control benchmark.

The gradient training here exists solely to quantify "how much does staying
gradient-free cost" against the HL rule-based controllers. It MUST NOT leak into
the HL mainline (`controllers/baseline_v*.py` stay gradient-free). See
openspec change `hl-lander-jax-rl-baseline` (design Decision 4).

Deliberately import-side-effect-free: training entrypoints import submodules
explicitly so that merely importing the package costs nothing.
"""
