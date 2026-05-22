## Why

`hl-lunar-lander-scaffold` change 已經把 `heuristic-learning/` 改成自有 `pyproject.toml` + `.venv`（JAX-only，禁 TF/PyTorch），原因是火力展示需要本子專案能單獨 clone / 單獨跑。但 `hl-research-roadmap` meta spec 仍寫著「所有 Python 程式 MUST 使用 `../learn-jax/.venv`、不得在 `heuristic-learning/` 內建立另一個 venv」——這條 requirement 現在與實作直接衝突。本 change 把 meta spec 對齊現況，並把「JAX-only 紅線」從單一 capability 提升到 repo 級契約，讓後續所有 `hl-*` capability 都自動繼承。

## What Changes

- **MODIFIED** `hl-research-roadmap` 的「共用 Python 環境契約」requirement → 改名為「獨立 venv 與 JAX-only 生態系契約」，內容反轉：本子專案 SHALL 自有 venv、套件加到 `heuristic-learning/pyproject.toml`、禁 `tensorflow*`（含 `tensorflow-probability`）與 `torch*`。
- 把 JAX-only 紅線（含 `distrax` 因 transitive 拉 TFP 而排除的已知地雷）提升為 repo 級規則。
- 不動其他三條 requirement（研究範圍、命名規範、實驗紀錄）。

## Capabilities

### New Capabilities
<!-- 無新 capability -->

### Modified Capabilities
- `hl-research-roadmap`: 「共用 Python 環境契約」requirement 反轉為「獨立 venv + JAX-only」；其餘 requirement 不變。

## Impact

- **Spec**：`openspec/specs/hl-research-roadmap/spec.md` 的一條 requirement 被取代（archive 本 change 時生效）。
- **程式碼**：無——實作已在 `hl-lunar-lander-scaffold` 完成；本 change 純粹讓 spec 追上現實。
- **文件**：`heuristic-learning/notes/env-contract.md`、`notes/roadmap.md` 已在前一個 change 標註契約變更，本 change archive 後可移除「歷史紀錄」字樣的暫時性。
- **後續 capability**：`hl-cv-heuristics`、`hl-mpc`、`hl-cpg`、`hl-lander-jax-rl-baseline` 等都將繼承 repo 級 JAX-only 契約，不需各自重述。
