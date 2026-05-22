## 1. 專案骨架

- [x] 1.1 建立 `heuristic-learning/` 資料夾與 `src/`、`experiments/`、`notes/` 子目錄
- [x] 1.2 撰寫 `README.md`（研究動機、目錄結構、OpenSpec 工作流說明）
- [x] 1.3 加入 `.gitignore`、`.python-version`、`Makefile doctor/list/view` 目標

## 2. 共用 Python 環境

- [x] 2.1 建立 symlink `./.venv -> ../learn-jax/.venv`
- [x] 2.2 驗證 `./.venv/bin/python -c "import jax, flax"` 成功
- [x] 2.3 在 `notes/env-contract.md` 紀錄「新依賴必須加到 learn-jax/pyproject.toml」的契約

## 3. OpenSpec 初始化

- [x] 3.1 `openspec init . --tools claude`
- [x] 3.2 建立此 change `bootstrap-hl-research` 的 proposal / tasks / spec
- [x] 3.3 `openspec validate bootstrap-hl-research --strict` 通過
- [ ] 3.4 後續 archive：`openspec archive bootstrap-hl-research`（人工觸發，不在本 run 完成）

## 4. Roadmap 文件

- [x] 4.1 在 `notes/roadmap.md` 列出 5 條後續 capability 的順序與相依性
- [x] 4.2 對每條 capability 寫一句話的「成功定義」，作為日後 proposal 的種子
