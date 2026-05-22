## Context

`hl-research-roadmap` 是 repo 級 meta spec，定義研究範圍、命名規範、環境契約、實驗紀錄四條 requirement。其中「共用 Python 環境契約」原本要求所有 Python 走 `../learn-jax/.venv`。`hl-lunar-lander-scaffold` change 已實際把 `heuristic-learning/` 改成自有 venv（JAX-only），meta spec 因此落後於現實。本 change 是純 spec 對齊，不含程式碼變動。

## Goals / Non-Goals

**Goals:**

- 讓 meta spec 的環境契約反映「自有 venv + JAX-only」的現況。
- 把 JAX-only 紅線（禁 TF/PyTorch、`distrax` 因 TFP 排除）提升為 repo 級規則，避免每個 capability 重述。

**Non-Goals:**

- 不動研究範圍、命名規範、實驗紀錄三條 requirement。
- 不改任何程式碼（實作已完成於前一個 change）。
- 不處理 `learn-jax/` 子專案自身的環境（那是另一個 repo 區塊）。

## Decisions

### Decision 1: 用 REMOVED（舊「共用」契約）+ ADDED（新「獨立 venv」契約）

OpenSpec 的 delta 以 requirement 標頭文字做比對：MODIFIED 要求標頭不變。本 change 的契約名稱要從「共用 Python 環境契約」反轉成「獨立 venv 與 JAX-only 生態系契約」，標頭改變，MODIFIED 會比對不上。因此採 REMOVED 舊條（附 Reason + Migration）+ ADDED 新條，語意最明確、archive 時不會 silently 失敗。

**Alternatives 考量：**

- MODIFIED 同名：要保留舊標頭「共用 Python 環境契約」，但新內容是「禁止共用、改自有 venv」，名實不符、誤導讀者。淘汰。
- RENAMED：只改名不改內容——但這裡內容整個反轉，RENAMED 不適用。淘汰。

### Decision 2: JAX-only 紅線寫進 meta spec，而非留在 capability spec

`hl-lunar-lander` capability spec 已有一條「獨立 venv 與 JAX-only」requirement。把它提升到 meta spec 後，capability spec 的那條變成「重述 repo 級規則」。保留 capability 那條無妨（更具體、可自我驗證），但 meta spec 必須是 single source of truth。

## Risks / Trade-offs

- **[風險] meta spec 與 capability spec 出現重複規則，未來不同步**　→　**緩解**：capability spec 那條明文標註「沿用 meta spec 的 repo 級契約」，meta spec 為唯一真實來源。
- **[Trade-off] `learn-jax` 與 `heuristic-learning` 從此各自維護 lockfile**：失去單一 lockfile 的維護便利，換得本子專案的可移植性（demo 訴求）。已在前一個 change 接受此取捨。
