PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: doctor list view propose-help hl-lander-deps-check hl-lander-smoke \
        hl-lander-eval hl-lander-watch hl-lander-train test

CONTROLLER ?= baseline_v1
SEEDS      ?= 5
EPISODES   ?= 10
SEED       ?= 0
STEPS      ?= 2000000

doctor:
	@echo "Python: $$($(PY) -c 'import sys; print(sys.executable)')"
	@$(PY) -c "import jax, flax; print('jax', jax.__version__, '/ flax', flax.__version__)"
	@command -v openspec >/dev/null && openspec --version || echo "openspec NOT installed"

# JAX-only red line: fail if any TensorFlow / PyTorch package is installed.
hl-lander-deps-check:
	@if $(PY) -m pip list 2>/dev/null | grep -iE 'tensorflow|^torch|torchvision|torchaudio|pytorch-lightning'; then \
		echo "RED LINE VIOLATION: TF/PyTorch package present in venv"; exit 1; \
	else \
		echo "deps-check OK: no tensorflow/torch packages"; \
	fi

hl-lander-smoke: hl-lander-deps-check
	$(PY) experiments/hl-lunar-lander/run.py --controller baseline_v1 --seeds 5 --episodes 10

# make hl-lander-eval [CONTROLLER=fsm_macro_v1] [SEEDS=5] [EPISODES=10]
hl-lander-eval: hl-lander-deps-check
	$(PY) experiments/hl-lunar-lander/run.py --controller $(CONTROLLER) --seeds $(SEEDS) --episodes $(EPISODES)

# make hl-lander-watch [CONTROLLER=fsm_macro_v1] [EPISODES=3] [SEED=0]
# CONTROLLER=rl_ppo → watch_rl.py；其他 → play_gui.py
hl-lander-watch:
	@if [ "$(CONTROLLER)" = "rl_ppo" ]; then \
		$(PY) experiments/hl-lunar-lander/watch_rl.py --episodes $(EPISODES) --seed $(SEED); \
	else \
		$(PY) experiments/hl-lunar-lander/play_gui.py --controller $(CONTROLLER) --episodes $(EPISODES) --seed $(SEED); \
	fi

# make hl-lander-train [STEPS=2000000] [SEED=0]
hl-lander-train:
	$(PY) experiments/hl-lunar-lander/train_rl.py --total-steps $(STEPS) --seed $(SEED)

test:
	$(PY) -m pytest

list:
	openspec list
	@echo "--- specs ---"
	openspec list --specs

view:
	openspec view

propose-help:
	@echo "In Claude Code: /opsx:propose \"<your idea>\""
	@echo "Then:           /opsx:apply  and  /opsx:archive"
