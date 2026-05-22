## 1. Spec 對齊（archive 時生效）

- [x] 1.1 確認本 change 的 delta spec（REMOVED 舊「共用 Python 環境契約」+ ADDED「獨立 venv 與 JAX-only 生態系契約」）通過 `openspec validate hl-roadmap-drop-shared-venv --strict`。
- [ ] 1.2 archive 本 change，使 `openspec/specs/hl-research-roadmap/spec.md` 的環境契約 requirement 被取代為新版。（透過 `/opsx:archive` 執行，非 apply 步驟）

## 2. 文件清理（archive 後）

- [x] 2.1 在 `heuristic-learning/notes/env-contract.md` 把「以下『複用 learn-jax』段落為歷史紀錄」的舊內容整段移除或壓縮成一行歷史註腳（契約已正式寫進 meta spec，不需要在 note 維持兩份）。
- [x] 2.2 在 `heuristic-learning/notes/roadmap.md` 把「需另開 change `hl-roadmap-drop-shared-venv`」這句標記為已完成（或移除），因為契約已對齊。

## 3. 驗收

- [ ] 3.1 用 `openspec list --specs` 確認 `hl-research-roadmap` 仍存在且只有環境契約那條被更新，其餘三條 requirement 不變。（內容替換在 archive 後生效；archive 前 `list --specs` 仍顯示 4 條、舊「共用」requirement 仍在）
