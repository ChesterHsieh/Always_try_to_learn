## Context

repo 已落地 HL（啟發式學習）骨架：`hl-research-roadmap`（meta：JAX-only、獨立 venv、capability 命名與實驗紀錄規範）與 `hl-lunar-lander`（`HeuristicPolicy` Protocol、runner、`BaselineLanderV1` 三段規則、弱對照組、controller 不可變性）。現況的程序化策略只存在於 `baseline_v1.py` 單檔，是一坨依高度／速度／角度排序的 if-else，無法被別的環境重用，也難以表達「狀態」與「多步動作」這兩種原文點名的構造（state machine、macro-action）。

本設計要做的，是把原文 procedural-policy 主題的三類積木——**規則表、有限狀態機、巨集動作**——抽成一個與環境無關的 `hl_core` 套件，並用一個 LunarLander 上的示範 controller 證明它能組裝出可迭代、可解釋、可回歸測試的策略。

關鍵約束（繼承自既有 spec，不重新發明）：
- **HL 紅線**：策略由可讀程式碼分支構成，禁止梯度／神經網路；本 capability 連 MPC / CPG 都不做。
- **JAX-only venv**：但本層為純規則邏輯，只用標準庫 + `numpy`，不新增任何相依。
- **`HeuristicPolicy` 介面神聖不可侵犯**：只有 `reset` / `act`，不得為了 trace 或組合而加 `update`/`learn`/`train`。
- **controller 不可變性**：示範 controller 一律新檔，絕不改 `baseline_v1.py`。
- **語言**：所有寫入 repo 的 Markdown 與 docstring 對齊既有 spec，使用繁體中文／英文混排的既有風格。

## Goals / Non-Goals

**Goals:**
- 提供與環境解耦、可被任意 `hl-*` import 的程序化策略積木：`RuleTable`、`FiniteStateMachine`、`MacroAction`。
- 提供 `ProceduralPolicy` 組合層，實作既有 `HeuristicPolicy`，並額外導出**唯讀**決策軌跡（state / 觸發規則 / macro 狀態 / action）。
- 在 LunarLander 上以 `FsmMacroLanderV1` 示範組裝，mean return ≥ `baseline_v1`，且 ≫ 弱對照。
- 把行為冻結為單元測試 + golden trace，落實原文「舊能力轉回歸測試」。

**Non-Goals:**
- 不做 MPC（滾動視窗最佳化）、CPG（相位振盪器）、CV heuristics——各自另開 capability。
- 不修改 `HeuristicPolicy` 介面、不改 `baseline_v1.py`、不改 runner 既有 dispatch 以外的行為。
- 不追求 `fsm_macro_v1` 是最強 LunarLander 策略；它只需證明「程式碼組合可迭代進步」。
- 不引入任何學習／參數搜尋；積木的所有「參數」皆為原始碼內的具名常數。

## Decisions

### D1：積木為純函式 + frozen dataclass，狀態與行為分離

`RuleTable`、`MacroAction` 的核心決策（guard / action_fn / 序列）為**無副作用純函式**，積木本身用 `@dataclass(frozen=True)` 持有不可變設定（規則清單、序列、優先序）。任何「執行進度」（FSM 當前 state、macro 已走幾步）不存在 frozen 設定物件裡，而是放在 `ProceduralPolicy` 持有的一個可重置的 runtime 狀態物件，由 `reset(seed)` 清空。

- **為何**：對齊全域 coding-style 的 immutability 紅線與原文「explicit readable variables」。設定不可變 → 規則表可被多個 policy 安全共享；進度集中在一處 → `reset` 一次清乾淨，確定性可重現（spec 要求同 seed 同 trajectory）。
- **替代方案**：把進度直接塞進積木物件（OOP 常見作法）。否決，因為會讓「設定」與「狀態」混在同一個可變物件，reset 容易漏欄位，破壞重現性——這正是既有 `policy.py` docstring 警告的坑。

### D2：以組合（composition）而非繼承串接三類積木

`FiniteStateMachine` 的每個 state 綁定一個**子策略**，子策略型別為一個窄 Protocol `Decider`（`decide(obs, ctx) -> (action, trace_record)`）。`RuleTable` 與「macro 包一層」都實作 `Decider`。`ProceduralPolicy` 持有最外層 decider（通常是 FSM），逐步呼叫並收集 trace。

- **為何**：composition 讓「FSM 的某個 state 用規則表、另一個 state 用 macro」變成自然組合，不需要繼承爆炸。`Decider` Protocol 與既有 `HeuristicPolicy` 用 `Protocol` 的風格一致（duck typing，見 `policy.py`）。
- **替代方案**：讓 FSM 直接 `isinstance` 判斷子策略是 RuleTable 還是 Macro。否決，違反開放封閉；新增第四類積木就得改 FSM。

### D3：trace 是決策的副產品，而非事後重建

每個 `Decider.decide` 回傳 `(action, TraceRecord)`，`TraceRecord` 是 frozen dataclass（步序、state 名、觸發規則名、macro 狀態、action）。`ProceduralPolicy.act` 內部呼叫 decider 拿到 `(action, record)`，把 record append 進唯讀 trace buffer，只回傳 `action` 給 runner。`decision_trace()` 回傳 buffer 的不可變快照（tuple）。

- **為何**：trace 與決策在**同一次呼叫**產生 → 不可能與真實行為不一致（spec 要求 trace 與行為一致、且導出無副作用）。回傳快照而非內部 list → 外部拿不到可變參照，杜絕「導出後被改」。
- **替代方案**：act 只回 action，trace 靠另一條 `explain(obs)` 重跑決策重建。否決，重跑可能與真實路徑分歧（例如 RNG、或讀到不同 ctx），且違反「導出無副作用」。

### D4：golden trace 用固定 seed 的 `decision_trace()` 序列化為 JSON

示範 controller 穩定後，跑固定 seed（例如 seed=0 的單一 episode），把 `decision_trace()` 序列化成 `tests/hl_core/golden/fsm_macro_v1.seed0.json`，回歸測試 load 後逐 record 比對。

- **為何**：trace 是結構化、確定性的，JSON diff 人類可讀，符合原文「golden traces」與「version diffs」。比對失敗的 PR 必須顯式更新 golden 並說明行為變更（spec 守門 scenario）。
- **替代方案**：比對最終分數而非逐步 trace。否決，分數相同不代表決策路徑相同（可能換了完全不同的規則卻巧合同分），無法防策略邏輯悄悄退化。

### D5：runner 整合最小化——只擴 `--controller` dispatch

`fsm_macro_v1` 透過既有 `run.py --controller fsm_macro_v1` 進入，dispatch 表新增一筆對應 `FsmMacroLanderV1`。runner 對 policy 一律只用 `HeuristicPolicy` 介面，不需要知道 trace 的存在。trace 與 golden 的產生走獨立的測試／實驗腳本，不污染 runner 主流程。

- **為何**：spec 明訂 runner 不需知道 `ProceduralPolicy` 存在即可運作。把 trace 導出留在測試層 → runner 行為對所有 controller 一致。
- **替代方案**：runner 每跑完就自動 dump trace。否決，會讓 runner 對特定 controller 型別有特殊待遇，違反介面一致性。

### 模組佈局

```
heuristic-learning/src/hl_core/
├── __init__.py        # 匯出 RuleTable / FiniteStateMachine / MacroAction / ProceduralPolicy / TraceRecord
├── rules.py           # RuleTable + Rule(guard, action_fn, priority, name)
├── fsm.py             # FiniteStateMachine + State + Transition
├── macros.py          # MacroAction（定長 / 帶 interrupt）
├── trace.py           # TraceRecord（frozen）、Decider Protocol
└── policy.py          # ProceduralPolicy（實作 HeuristicPolicy）+ runtime 狀態物件

heuristic-learning/src/hl_lander/controllers/
└── fsm_macro_v1.py    # FsmMacroLanderV1：用 hl_core 積木組裝 descend→align→touchdown + touchdown macro

heuristic-learning/tests/hl_core/
├── test_rules.py / test_fsm.py / test_macros.py / test_policy_trace.py
└── golden/fsm_macro_v1.seed0.json
```

## Risks / Trade-offs

- **[過度設計：三類積木對 LunarLander 可能殺雞用牛刀] →** 接受。本 capability 的價值在於「可被後續所有 `hl-*` 重用的共用層」，LunarLander 只是首個驗證載體；以最小示範（三 state + 一 macro）證明可組合即可，不強求複雜度。
- **[`fsm_macro_v1` mean return 可能無法 ≥ baseline_v1] →** Mitigation：先以 `baseline_v1` 的三段規則為 descend/align 兩個 state 的 `RuleTable` 起點（行為等價），再僅在 touchdown 階段加 macro 微調；如此下界至少持平，macro 提供向上空間。若仍不達標，spec 已定義「視為 bug 必須調查」而非放寬門檻。
- **[golden trace 太脆，無關緊要的重構也會誤觸] →** Mitigation：golden 只比對語意欄位（state 名、規則名、action、macro active 布林），不比對浮點觀測值或時間戳；純重構（改名內部變數、補 type hint）不應改變這些語意欄位。
- **[trace buffer 在長 episode 累積記憶體] →** 接受。LunarLander episode 上限數百步，trace record 為小 dataclass，量級可忽略；若未來長視窗環境需要，再另議上限或抽樣策略（屬 `hl-state-graph-search` 範圍）。
- **[依賴方向被意外反轉（hl_core import hl_lander）] →** Mitigation：spec 已有 scenario 守門，且 `hl_core` 測試獨立於 `hl_lander`，import 反轉會讓 `hl_core` 測試在沒有 lander 的情境下失敗而被發現。

## Migration Plan

無資料遷移、無破壞性變更。落地步驟：
1. 新增 `hl_core` 套件與單元測試（純新增，不碰既有檔）。
2. 新增 `fsm_macro_v1.py` 並在 runner dispatch 表加一筆（既有 controller 行為不變）。
3. 跑四方對照（noop / random / baseline_v1 / fsm_macro_v1），寫 REPORT.md 新 section。
4. 凍結 golden trace、納入回歸測試。

回滾：移除 `hl_core/`、`fsm_macro_v1.py`、dispatch 那一筆與 golden 檔即可，既有 capability 完全不受影響。

## Open Questions

- `ctx`（傳給 guard / action_fn / condition 的上下文）要放哪些欄位？初版傾向最小化：只放 `step_index` 與上一個 action；若 align 階段需要「連續 N 步角度穩定」這類跨步判斷，再擴 `ctx` 而非讓積木持有可變狀態。實作時於 design 範圍內定案，無需另開 change。
- golden trace 的固定 seed 選 0 或多 seed？初版用單 seed=0 控制脆度；若日後發現單 seed 漏抓退化，再加 seed（屬測試強化，不改 spec）。
