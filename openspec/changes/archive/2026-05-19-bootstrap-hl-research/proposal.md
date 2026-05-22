## Why

[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh) 提出「啟發式學習（HL）」典範：不訓練神經網路，而是讓 agent 在環境反饋下迭代手寫規則與程式化策略。這個典範跨越多個子領域（程序化策略、MPC、CPG、state-graph search、CV heuristics），需要一個能延展的研究專案結構，而不是把所有 code 塞進單一資料夾。

導入 OpenSpec / SDD 的原因：
- 每一個子主題（如 MPC for Ant）都是一條獨立的能力線，適合用 capability spec 描述「應該能做什麼」，與實作脫鉤。
- 之後加入新方法時，可以審視既有 spec 的 delta，避免無紀錄改寫。
- 與 `learn-jax`、`mlops-zoomcamp` 等姊妹專案共用一致的 SDD 紀律。

## What Changes

本變更只負責「立案」這個專案的研究骨架，**不**包含任何演算法實作。內容：

- 在 `openspec/specs/` 下建立第一個 capability：`hl-research-roadmap`，紀錄此資料夾的研究範圍、共用環境契約、與後續 capability 的命名規範。
- 規劃後續 5 條 capability 線（不在本 change 範圍內，列為 Out of Scope）：
  - `hl-procedural-policy`：rule-based / state machine 控制器
  - `hl-mpc`：short-horizon model predictive control
  - `hl-cpg`：central pattern generator gait
  - `hl-state-graph-search`：long-horizon graph 規劃
  - `hl-cv-heuristics`：edge / morphology / connected components 視覺前處理

- 約束共用 Python 環境：所有實驗必須使用 `../learn-jax/.venv`（透過 `./.venv` symlink）；新增依賴一律寫到 `learn-jax/pyproject.toml`。

## Capabilities

### New Capabilities

- `hl-research-roadmap`: 定義此 repo 的研究範圍、capability 命名規範、共用 Python 環境契約與實驗紀錄方式。

### Modified Capabilities

（無，這是首個 change）

## Impact

- 新增資料夾：`heuristic-learning/{src,experiments,notes}/`
- 新增檔案：`README.md`、`Makefile`、`.python-version`、`.gitignore`
- 新增 symlink：`heuristic-learning/.venv -> ../learn-jax/.venv`
- 依賴：透過 `learn-jax/pyproject.toml`（JAX 0.4.38、flax 0.10.4 已就緒；OpenCV / envpool / mujoco / vizdoom 在後續 capability 提案時再加入）
- 工具鏈：需要 Node.js ≥ 20.19 來執行 `openspec` CLI；本機已安裝 v24.3.0。

## Out of Scope

- 任何 HL 演算法的實作（留給 `hl-procedural-policy` 等後續 change）。
- 環境 wrapper（envpool、mujoco、vizdoom）：在第一個需要它們的 capability change 才安裝。
- Benchmark / Atari57 完整重現：屬於進階階段，最後再提案。
