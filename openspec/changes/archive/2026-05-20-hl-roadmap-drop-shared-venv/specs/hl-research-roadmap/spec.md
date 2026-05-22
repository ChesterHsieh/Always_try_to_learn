## REMOVED Requirements

### Requirement: 共用 Python 環境契約
**Reason**: 火力展示需要 `heuristic-learning/` 能單獨 clone / 單獨跑，依賴 sibling 目錄 `../learn-jax/.venv` 與此目標衝突。`hl-lunar-lander-scaffold` change 已實際改成自有 venv。
**Migration**: 不再使用 `../learn-jax/.venv` 與根目錄 `./.venv` symlink。改用 `heuristic-learning/pyproject.toml` + `uv venv .venv && uv sync` 自建 venv，套件加到本子專案的 `pyproject.toml`。詳見新的「獨立 venv 與 JAX-only 生態系契約」requirement。

## ADDED Requirements

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
