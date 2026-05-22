## 1. 環境與依賴

- [x] 1.1 在 `heuristic-learning/` 內建立獨立 `pyproject.toml`（用 `uv init` 或手寫），dependencies 列出 JAX 生態系核心：`jax`、`jaxlib`、`flax`、`optax`、`gymnax`、`gymnasium[box2d]>=1.0`、`numpy`（**不含 `distrax`**，因其 transitive 拉進 tensorflow-probability）；移除 `.venv` symlink，改成 `uv venv heuristic-learning/.venv && uv sync` 在本 repo 自建 venv（macOS 若 Box2D 安裝失敗，先 `brew install swig` 再重跑）。
- [x] 1.2 驗證 `heuristic-learning/.venv/bin/uv pip list` 不包含 `tensorflow`、`tensorflow-*`、`torch`、`torchvision`、`torchaudio`、`pytorch-lightning`；若有 transitive 拉進，回頭調整 dependencies（例如改用 extras-free 變體）直到 list 乾淨。把這條檢查指令寫進 `Makefile` 的 `hl-lander-deps-check` target，供 CI 反向 assertion。
- [x] 1.3 確認 `heuristic-learning/.venv/bin/python -c "import jax, gymnasium; gymnasium.make('LunarLander-v3')"` 可成功。（jax 0.4.38 / gymnasium 1.3.0 / LunarLander-v3 OK）
- [x] 1.4 更新 `heuristic-learning/README.md` 的「環境」段落：說明已改成自有 venv、列出 JAX-only 紅線（禁 TF/PyTorch）、補上 `swig` 前置安裝；同步在 `heuristic-learning/notes/env-contract.md` 加 note 註記「本子專案已脫離 learn-jax 共用 venv 契約，原因 = demo 移植性」。

## 2. 套件骨架

- [x] 2.1 建立 `heuristic-learning/src/hl_lander/__init__.py`（空 package marker；不在 `__init__` 內做副作用 import）。
- [x] 2.2 建立 `heuristic-learning/src/hl_lander/policy.py`，定義 `HeuristicPolicy`（建議用 `typing.Protocol`，僅 `reset(seed: int) -> None` + `act(observation: np.ndarray) -> int`），並加上 docstring 引用本 capability spec。
- [x] 2.3 建立 `heuristic-learning/src/hl_lander/env.py`，提供 `make_env(seed: int) -> gym.Env`，內部固定 env id `LunarLander-v3` 並處理 seed/reset 規格；不暴露其他 helper。
- [x] 2.4 建立 `heuristic-learning/src/hl_lander/runner.py`，提供 `run_episode(env, policy, seed) -> EpisodeResult`（記錄 return、length、landed 與否）以及 `evaluate(policy_factory, seeds, episodes_per_seed) -> List[EpisodeResult]`。
- [x] 2.5 建立 `heuristic-learning/src/hl_lander/metrics.py`，提供 `summarize(results) -> {mean_return, std_return, landing_rate, n}` 純函式（無 IO）。
- [x] 2.6 建立 `heuristic-learning/src/hl_lander/controllers/__init__.py`（空 package marker）。

## 3. Baseline Controller v1

- [x] 3.1 在 `heuristic-learning/src/hl_lander/controllers/baseline_v1.py` 實作 `BaselineLanderV1`，遵守 `HeuristicPolicy`；內部以 obs[4]（angle）、obs[5]（angular velocity）、obs[1]（y position）、obs[3]（y velocity）做三段式規則，輸出 0/1/2/3 離散動作。
- [x] 3.2 為 `BaselineLanderV1.reset` 寫明「無內部狀態時 no-op」的 docstring 並確保多次 reset 不會洩漏 RNG 狀態。
- [x] 3.3 在 `baseline_v1.py` 檔頭加 docstring 註明「v1 為 HL 迭代軌跡的起點；合入 main 後不可變，後續迭代開 `baseline_v2.py` 而非就地修改」。同時在 `controllers/__init__.py` 加一段註解陳述此版本不可變政策。

## 4. 實驗入口與紀錄

- [x] 4.1 建立 `heuristic-learning/experiments/hl-lunar-lander/run.py`，CLI 介面：`--controller {baseline_v1|random|noop}`（以檔名為 key dispatch、不允許 `baseline` 這種無版號別名）、`--seeds`、`--episodes`、`--render`（預設關），呼叫 `runner.evaluate` + `metrics.summarize`，把結果以 `## {controller-name} ({YYYY-MM-DD})` 形式的 section append（不覆寫其他 section）到 `experiments/hl-lunar-lander/REPORT.md`。執行時 MUST 透過 `subprocess` 取 `git rev-parse HEAD` 寫入該 section。
- [x] 4.2 建立 `heuristic-learning/experiments/hl-lunar-lander/REPORT.md` 初始檔，結構：
  - 第一段「版本不可變性說明」：陳述「每個 `baseline_v{n}` 是一個獨立 first-class artefact，合入 main 即不可變」。
  - 第二段「對照組」placeholder：列出 noop / random / baseline_v1 三個 section 標頭與待填欄位（執行指令、gymnasium 版本、env id、seed 列表、mean ± std、landing rate、執行日期、git commit hash）。
  - 第三段「RL 對照組（未來工作）」：標註會由 `hl-lander-jax-rl-baseline` change 處理，暫以「文獻 PPO ~280、DQN ~250」為口頭對照。

## 5. Make Target 與 Smoke Test

- [x] 5.1 在 `heuristic-learning/Makefile` 新增 `hl-lander-smoke` target：`.venv/bin/python experiments/hl-lunar-lander/run.py --controller baseline_v1 --seeds 5 --episodes 10`，並把它列進 `.PHONY`；同時把 1.2 的 `hl-lander-deps-check` 列為 smoke 的前置依賴（make hl-lander-smoke 應先跑 deps-check）。
- [x] 5.2 確認 smoke target 在本地 CPU 60 秒內跑完，mean return ≥ 0；若不過，**開新檔案 `baseline_v2.py` 修規則**（不就地改 v1、不調 hyperparameter，改寫程式碼是 HL 的學習方式）——但 v2 的引入應屬於後續 change，本 change 在 baseline_v1 不過時應停下並回頭檢查 runner / env 是否有 bug。

## 6. 驗收與紀錄

- [x] 6.1 跑一次完整 smoke，把實際 mean / std / landing rate 填回 `REPORT.md`，附上實際指令與 `gymnasium.__version__`。
- [x] 6.2 用 `openspec validate hl-lunar-lander-scaffold --strict` 驗證所有 artifacts 過關。
- [x] 6.3 在 `heuristic-learning/notes/roadmap.md` 註記：第一個 capability 已落地，後續 `hl-cv-heuristics`、`hl-mpc`、`hl-cpg` 可以 fork 這套骨架。
- [x] 6.4 在 archive 本 change 前，另開一個新 change（建議名稱 `hl-roadmap-drop-shared-venv`）修改 `hl-research-roadmap` spec 的「共用 Python 環境契約」requirement，使其不再強制共用 `learn-jax/.venv`，並把 JAX-only 紅線提升到 meta spec 層級。（已於 2026-05-20 建立該 change）

## 7. 弱對照組與版本管理

- [x] 7.1 建立 `heuristic-learning/src/hl_lander/controllers/random.py`，實作 `RandomLander`：`reset(seed)` 用 `numpy.random.default_rng(seed)` 建立內部 RNG，`act(obs)` 忽略 obs、回傳 `int(rng.integers(0, 4))`。
- [x] 7.2 建立 `heuristic-learning/src/hl_lander/controllers/noop.py`，實作 `NoOpLander`：`reset(seed)` no-op、`act(obs)` 永遠回傳 `0`。
- [x] 7.3 在 `run.py` 加上 controller dispatch（檔名為 key）：`baseline_v1` → `controllers.baseline_v1.BaselineLanderV1`、`random` → `controllers.random.RandomLander`、`noop` → `controllers.noop.NoOpLander`。不接受沒有版號的 `baseline` 別名。
- [x] 7.4 跑一次三方對照（依序 `noop`、`random`、`baseline_v1` 各 5 seeds × 10 episodes），把三段結果 append 到 REPORT.md，並驗證觀察到 noop < random < baseline_v1 的排序；若違反，先檢查 runner 與 controller 實作，不要為了「拿到好看數字」就改實驗設定。
- [x] 7.5 開新 change `hl-lander-jax-rl-baseline` 的草稿位置（先不實作，只在 roadmap 留 stub）：在 `heuristic-learning/notes/roadmap.md` 加一節「RL 對照組（待開新 change）」，說明會用 JAX 手寫 minimal PPO/DQN（300–500 行）、與本 change 的 gymnasium env 共用、不引入 PyTorch。
