---
project: heuristic-learning
---

# hl-research-roadmap Specification

## Purpose

定義此 repo 的研究範圍、capability 命名規範、獨立 venv 與 JAX-only 生態系契約與實驗紀錄方式，使後續所有 HL（啟發式學習）子主題能在一致的骨架下展開。

## Requirements

### Requirement: 研究範圍宣告
此 repo SHALL 以「Learning Beyond Gradients」一文的啟發式學習典範為研究範圍，並 MUST 排除任何以梯度下降訓練神經網路為主軸的方法。

#### Scenario: 新增 capability 時的範圍檢查
- **WHEN** 任何新的 OpenSpec change 被提出
- **THEN** 該 change 的 `proposal.md` 必須在 `## Why` 區段內，明確說明它與 HL 典範（rule-based / MPC / CPG / search / CV heuristics 之一）的關聯
- **AND** 若涉及梯度訓練，必須在 `## Out of Scope` 或拒絕該 change

### Requirement: 獨立 venv 與 JAX-only 生態系契約
`heuristic-learning/` SHALL 擁有自己的 `pyproject.toml` 與獨立 venv（路徑 `heuristic-learning/.venv`），以 `uv` 管理。本 repo 內所有 Python 程式 MUST 透過該 venv 執行，且其 dependencies MUST 僅依賴 JAX 生態系（jax、jaxlib、flax、optax、gymnax、gymnasium 等），MUST NOT 直接或間接依賴 `tensorflow`、`tensorflow-*`（含 `tensorflow-probability`）、`torch`、`torchvision`、`torchaudio`、`pytorch-lightning` 等 TF / PyTorch 套件。

#### Scenario: 執行任何 Python 程式
- **WHEN** 開發者或 CI 執行此 repo 內的 Python 程式
- **THEN** 直譯器 MUST 是 `heuristic-learning/.venv/bin/python`，不得使用系統 Python、`../learn-jax/.venv` 或 conda 環境
- **AND** `python -c "import jax, flax, gymnasium"` 必須能成功 import

#### Scenario: 需要新增 Python 套件
- **WHEN** 一個 capability 需要新的 Python 套件（例如 `opencv-python`、`mujoco`、`envpool`、`vizdoom`）
- **THEN** 套件必須加到 `heuristic-learning/pyproject.toml` 的 `dependencies` 或 `dev-dependencies`，並以 `uv sync` 安裝
- **AND** 安裝後 MUST 驗證 `uv pip list` 不含任何 TF / PyTorch 套件；若某套件 transitive 拉進 TF/PyTorch（已知地雷：`distrax` 拉 `tensorflow-probability`），MUST 改用 extras-free 變體或拒絕加入

#### Scenario: 後續 capability 繼承本契約
- **WHEN** 任何新的 `hl-*` capability 被提案
- **THEN** 該 capability 預設繼承本 repo 級 JAX-only 契約，不需在自己的 spec 重述（可選擇重述以利自我驗證，但 meta spec 為唯一真實來源）

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
</content>
</invoke>