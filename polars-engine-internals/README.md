# Polars 引擎內部

追 Polars 原始碼，搞懂 1.x → 2.0 為什麼把 streaming 引擎變成 `LazyFrame.collect()` 的預設、為什麼峰值記憶體能掉那麼多，
並把它放進 CMU 15-445/721 資料庫課程的座標裡（執行模型、查詢最佳化、外部排序、hash join）。

## 內容

| 資料夾 | 內容 | 線上開啟 |
|---|---|---|
| [report/](report/) | 1.0 → 2.0 導讀報告：一行 `match` 的翻轉、記憶體為什麼小那麼多、新舊引擎架構、行順序與 OOC 的取捨、optimizer 的變化、決策因素。6 張圖，每段都附固定在 tag 的原始碼連結 | [報告](https://chesterhsieh.github.io/Always_try_to_learn/polars-engine-internals/report/) |
| [skill-tree/](skill-tree/) | 互動技能樹＋題庫：50 節點、11 章、153 題四選一，每題附解釋與可複製的深挖 prompt，答題正確率參與節點點亮 | [技能樹](https://chesterhsieh.github.io/Always_try_to_learn/polars-engine-internals/skill-tree/) |

建議順序：先讀報告建立全貌，再從技能樹的序章開始，邊讀原始碼邊做題。

## 對照的版本

| tag | commit | 日期 | 備註 |
|---|---|---|---|
| `py-1.0.0` | `f73937a` | 2024-07-01 | 預設 in-memory；`streaming=True` 走舊引擎 `polars-pipe`；`polars-stream` 只是 3.4k 行的骨架 |
| `py-2.0.0` | `22a147d` | 2026-10-06 | `Engine::Auto` 解析成 streaming（PR #27822）；`polars-pipe` 已在 1.31 刪除；新增 `polars-ooc` |

## 重建

```bash
cd skill-tree && python3 make.py   # 驗證 + 產生 polars-engine.html 與 index.html
```

報告是手寫的單檔 HTML（內嵌 SVG），直接改 `report/index.html` 即可。

## 限制

報告與題庫的結論來自原始碼閱讀與公開資料，沒有實際編譯或執行 Polars 做驗證；文中標為「推論」的段落是對程式碼的解讀。
join reordering、基數模型、runtime filter、OOC 預設開啟都在 2.0 tag 前六週內才合併，之後幾個版本很可能還會變。
