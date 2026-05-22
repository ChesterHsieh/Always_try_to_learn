## ADDED Requirements

### Requirement: 研究範圍宣告
此 repo SHALL 以「Learning Beyond Gradients」一文的啟發式學習典範為研究範圍，並 MUST 排除任何以梯度下降訓練神經網路為主軸的方法。

#### Scenario: 新增 capability 時的範圍檢查
- **WHEN** 任何新的 OpenSpec change 被提出
- **THEN** 該 change 的 `proposal.md` 必須在 `## Why` 區段內，明確說明它與 HL 典範（rule-based / MPC / CPG / search / CV heuristics 之一）的關聯
- **AND** 若涉及梯度訓練，必須在 `## Out of Scope` 或拒絕該 change

### Requirement: 共用 Python 環境契約
所有實驗與程式碼 MUST 使用 `../learn-jax/.venv` 作為 Python 直譯器來源，並 SHALL 透過本 repo 根目錄的 `./.venv` symlink 存取。

#### Scenario: 執行任何 Python 程式
- **WHEN** 開發者或 CI 執行此 repo 內的 Python 程式
- **THEN** 直譯器必須是 `./.venv/bin/python`
- **AND** `python -c "import jax, flax"` 必須能成功 import

#### Scenario: 需要新增 Python 套件
- **WHEN** 一個 capability 需要新的 Python 套件（例如 `opencv-python`、`mujoco`、`envpool`、`vizdoom`）
- **THEN** 套件必須加到 `../learn-jax/pyproject.toml` 的 `dependencies` 或 `dev-dependencies`
- **AND** 必須以 `uv sync`（或等效指令）安裝，**不得**在 `heuristic-learning/` 內建立另一個 venv

### Requirement: Capability 命名規範
所有 capability MUST 以 `hl-` 前綴開頭，並 SHALL 使用 kebab-case，對應「Learning Beyond Gradients」原文的子主題。

#### Scenario: 提案新的 capability
- **WHEN** 開發者透過 `/opsx:propose` 或手動建立新 capability
- **THEN** capability 名稱必須匹配正則 `^hl-[a-z0-9-]+$`
- **AND** 名稱必須能對應到下列其中一個原文主題：procedural-policy、mpc、cpg、state-graph-search、cv-heuristics，或是這些主題下進一步細分的子主題

### Requirement: 實驗紀錄結構
每一個對應 HL 子主題的實驗 MUST 在 `experiments/<capability-name>/` 下留下可重現的程式入口，並 SHALL 附上一份簡短報告。

#### Scenario: 完成一次實驗
- **WHEN** 一個 capability 的 `tasks.md` 中的某個 task 涉及「跑出一個結果」
- **THEN** 對應的 `experiments/<capability>/run.py`（或等效 entrypoint）必須存在
- **AND** `experiments/<capability>/REPORT.md` 必須包含：環境 seed、執行指令、預期分數、實際分數
