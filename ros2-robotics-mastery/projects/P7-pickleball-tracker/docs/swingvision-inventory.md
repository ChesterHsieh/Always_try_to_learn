# SwingVision 功能全覽（研究日期 2026-09-03）

> 用途：`pickleball_msgs` 介面必須裝得下這份清單的資訊。
> 圖例：**T** 網球限定、**P** pickleball 限定、**B** 兩者。⚠ = 官方未證實（第三方／推測／過時）。
> `swing.tennis` 已 301 到 `swing.vision`；舊 Zendesk 說明中心已關閉。

主要來源：
- 官網 https://swing.vision/ ｜ 定價 https://swing.vision/subscribe ｜ 指南 https://swing.vision/guides ｜ FAQ https://swing.vision/faq
- Swing Court https://swing.vision/swing-court ｜ Teams https://swing.vision/teams ｜ 電子報 https://swing.vision/newsletters
- App Store https://apps.apple.com/us/app/swingvision-tennis-pickleball/id989461317
- ITF PAT 核可報告 PAT-25-037 https://www.itftennis.com/media/15294/pat-25-037-approval-report-swingvision.pdf
- 評測：techinthesun.com、betterpickleball.com、tennis.com、tennisnerd.net、aubonetennis.com、secondservepodcast.com、developer.apple.com、acesense.io、speakpickleball.com

---

## 1. 擷取／設定

| 功能 | 說明 | 範圍 |
|---|---|---|
| 單後置鏡頭擷取 | 一支 iPhone/iPad 架在底線後，60 fps 1080p（Max 4K）；ML 在裝置上即時跑球、球場、球員追蹤 | B |
| 第二支手機（選配） | 另一端再放一支；Swing Court 有雙鏡頭 ELC「近線 99%」 | T（ITF 證實） |
| 支援裝置 | iPhone 11 / SE 2020 以上（A13+）、2020 iPad Pro 以上；iOS 18 才有即時統計與線審 | B |
| 三種架設模式 | Ground（純去死時間 + 擊球篩選）、Fence Mount（加球速/深度/穩定度 + 半場線審）、Swing Stick（全場線審） | B |
| 室內／室外設定 | 指南有分頁 | B |
| Swing Stick | 圍網夾桿約高出圍網 2 ft、遮陽、定位輔助；Pro/Max 年約附贈 | B |
| 廣角鏡頭配件 | 無超廣角裝置時用 | B |
| 架設指引 | 底線後、≥5 ft 高、兩條底線都入鏡、置中、太陽在背後、1x 變焦、網不遮遠端線 | B |
| 球場自動偵測／框位驗證（Auto Setup） | 手機（與 Watch）確認框位品質；球場關鍵點偵測 | B |
| Watch 相機預覽 | 確認對位 | B |
| Remote Control（QR 配對） | 任何瀏覽器掃 QR 遙控錄影與挑戰 | B |
| Watch 啟停 + Siri | Series 6/SE 以上 | B |
| Session 設定 | 運動、類型（Match/Practice/Rally/Serve Practice/Ball Machine）、單雙打、標記球員 | B |
| 即時處理 vs 匯入 | 即時裝置上處理；或匯入 GoPro/Android 影片（iOS 或 macOS 15+），匯入者無即時挑戰與 Watch 比分 | B |
| 連續錄影 | 不用重設定 | B ⚠ |
| Focus 自動化 | 避免來電中斷 | B |
| 過熱處理 | 語音提示關閉音訊回饋（v11.9.65） | B |
| 雲端備份 | Free 7 天，Pro/Max 永久 | B |
| ITF PAT 核可 | PAT-25-037（v11.9.6，2025-10）；Watch 不得在賽事使用 | T |
| Swing Court（固定安裝） | 每場 1 相機 + 1 kiosk，防水、Wi-Fi/乙太、ELC、直播、音訊回饋、排行榜 | B |

## 2. 球追蹤輸出

| 功能 | 說明 | 範圍 |
|---|---|---|
| 3D 擊球點 | 每拍記錄 | B（ITF） |
| 3D 觸網點 | 觸網時記錄 | B（ITF） |
| 3D 落點 | 每次彈跳；匯出欄位 `Bounce (x/y)`、`Hit Side` | B |
| 3D 軌跡 | 每拍軌跡；送到場邊平板做線審 | B |
| 球速 | 每軌跡平均速度（比雷達峰值低約 20%）；±10%；mph/km/h | B |
| 旋轉類型 | 網球：topspin/slice/flat；發球 kick/slice/flat。Pickleball 評測稱「不知道旋轉」 | T（P ⚠） |
| 旋轉 RPM | 部分評測提到（舊 Watch 陀螺儀時代） | T ⚠ |
| In/Out 線審 | 每次彈跳；10 cm 內 97%，整體 >99%；半場 vs 全場；低信心標 ⚠ | B |
| 落在網哪側 | 挑戰列表顯示 ↑/↓ | B |
| 擊球結果 | In/Out/Net；失誤分 net/long/wide | B |
| 深度 | 發球線後 %、深度區；發球/接發 short vs deep | B |
| 方向 | cross-court / down-the-line 等 | B |
| Net clearance | 只出現在教練建議，未證實為量測輸出 | ⚠ |
| 跟球裁切 | 匯出時球置中 | B |

## 3. 球員追蹤

| 功能 | 說明 | 範圍 |
|---|---|---|
| 球員偵測與身分 | ML 追蹤；使用者從縮圖標記；雙打支援 | B |
| 擊球位置 | 每拍站位；熱圖區：底線內 / 後 0–3 ft / >3 ft / 邊道；`Hit Zone`（Deuce/Ad） | B |
| 站位時間分布 | 底線後 / 底線到發球線 / 網前 % | T |
| 跑動量 | 「5 個最累回合」隱含每回合距離；官方未顯示距離數字 | B |
| 覆蓋熱圖 | 位置熱圖 | B ⚠ |
| 步法／技術 | 手動用篩選 + 慢動作看；無骨架輸出 | B ⚠ |
| 發球站位提示 | Pickleball Match Mode 告知每次發球站哪 | P |
| Watch 生理數據 | 心率、卡路里、距離 → HealthKit | B |

## 4. 單拍統計

| 功能 | 說明 | 範圍 |
|---|---|---|
| 擊球分類（網球） | Forehand/Backhand/Serve（ITF）；Volley/Overhead；Slice 由旋轉 | T |
| 擊球情境 | Serve（first/second）、Return、Serve+1、Return+1、finishing shot | B |
| Pickleball 擊球分類 | Dink、Drop、3rd Shot、Serve、Return、Volley（v11.9.46 改版）；Drive/Lob 未官方命名 | P |
| 發球側 | Deuce / Ad | B |
| 穩定度 % | 發球進球率（deuce/ad）、正反手進球率、整體 | B |
| 球速 | 各擊球平均速度、最快發球 | B |
| 深度與分布區 | 發球區後 %、中三分之一 %、左右比較 | B |
| 擊球分布圓餅 / 旋轉分布 | 依擊球種類 | T |
| Winner / UE / FE | Watch 或 AI 計分標記；Match 檢視依正反手拆 | B |
| 回合長度 | 每回合拍數、>5 拍 %、最長回合 | B |
| 每次差異 | 每項統計旁顯示與上次 % 變化 | B |
| 時間戳 | 每拍/回合/分數皆有 | B |
| 原始資料匯出 | Pro；XLSX：Settings/Shots/Points/Games/Sets/Stats；Shots 欄位：Player, Stroke, Type, Result, Hit Zone, Hit Side, Spin, Speed, Direction, Bounce(x/y) | B ⚠ |

## 5. 比賽統計

| 功能 | 說明 | 範圍 |
|---|---|---|
| 計分模式 | Final Score（AI 填點）、Game by Game、Point by Point、Point by Point+（含如何贏）；Watch 手勢、選發球者、復原、換邊提醒 | B |
| AI 計分 | 由終局比分自動推 point-by-point（Pro） | B |
| 分數編輯 | 改誰贏、怎麼贏 | B |
| Point-by-point 時間軸 | 每分時間軸；比分篩選 | B |
| 發球統計 | 一二發 %、發球得分、發球速度、雙誤、發球落點熱圖與失誤類型 | B |
| 接發統計 | 接發成功率、深淺、deuce/ad、球速、對一二發 | B |
| 破發點／盤末點 | 轉換率、挽救數 | T |
| 三種熱圖 | 落點、擊球位置、發球落點；可依擊球/進出/deuce-ad/旋轉篩選；切換對手 | B |
| 對手偵察 | 點對手看其統計 | B |
| Rally scoring 提醒 | 接發方凍結期提醒 | P ⚠ |
| 比賽時長／盤數／剩餘挑戰 | 場邊平板 UI | T |
| 長期趨勢 | 跨場次趨勢圖、週目標、篩選 | B |
| 影片比分 overlay | 匯出時渲染 | B |

## 6. 影片輸出

| 功能 | 說明 | 範圍 |
|---|---|---|
| 去死時間 | 2 小時 → 約 15 分鐘；Free 也有 | B |
| 按回合／按拍切段 | Point-by-Point / Rally-by-Rally / Shot-by-Shot | B |
| 影片篩選 | 球員、擊球、旋轉、擊球類型、方向、finishing shot、比分類型、贏家、回合長度、In/Out | B |
| 自動精華 | 5 個最長回合、5 個最累回合、>5 拍回合、AI 精選 | B |
| 收藏／排除 | Watch 即時收藏 | B |
| Overlay | 比分 + 統計 | B |
| 慢動作／逐幀 | Watch 數位錶冠逐幀、縮放 | B |
| 自動跟球直式輸出 | 攝影師式裁切 | B |
| 匯出限制 | Free ≤5 分鐘含 logo；Pro 無限 | B |
| 分享 | 網頁連結、IG/FB/WhatsApp/TikTok/YouTube、隱私設定 | B |
| 4K | Max | B |
| 直播 | Teams 每季 15 場、1080p 含比分板；Swing Court | B |

## 7. 即時

| 功能 | 說明 | 範圍 |
|---|---|---|
| 線審挑戰 | 挑戰最近 4 拍；Watch / 瀏覽器 / 場邊 iPad；顯示判定、擊球類型、網側、重播 | B |
| 語音 "Out!" | 2024-12 beta；每次出界喊 Out；Free 也有 | B |
| 即時比分板 | Watch、平板、直播 | B |
| Goal Mode / 目標區語音回饋 | 設目標區，Siri 播報命中率、球速、進度、回合長度 | B |
| 即時語音統計 | 分數與統計 | B ⚠ |
| Watch 運動 | 即時心率、卡路里 | B |

## 8. 教練／分析

| 功能 | 說明 | 範圍 |
|---|---|---|
| AI Coaching / Strategy Insights | 單打賽後：最佳發球與接發策略、最弱模式、點擊重播（Max）；雙打「之後推出」 | T（P ⚠） |
| 個人化賽後教練 + 週目標 | App Store 描述 | B |
| 趨勢 | 跨場次圖表 | B |
| 練習模式 | Practice、Rally、Serve Practice、Ball Machine + Goal Mode | B |
| 教練評閱市集 | 送給認證教練，7 天內回覆 | B |
| 遠端教練／標記 | 標記教練；Teams 儀表板 | B |
| 比較 | 與對手並排；與自己過去場次 | B |
| 社群 | 群聊、排行榜 | B |

## 9. Pickleball 專屬

| 功能 | 說明 |
|---|---|
| Pickleball 模式（v11.0，2023-11） | 單雙打；Rally Mode；Match Mode + Watch 計分（v11.6） |
| Dink / Drop / 3rd shot 偵測 | 統計、篩選、精華（2024-12） |
| 發球與接發深度 | short vs deep |
| 發球站位提示 | 每次發球站哪 |
| Rally-scoring 提醒 | ⚠ |
| Serve Practice / Ball Machine | v11.9.46 |
| 語音線審 | 需專用 pickleball 球場、所有線可見 |
| **Kitchen / NVZ 違例偵測** | **官方完全沒有** |
| Drive / lob / erne / ATP 分類 | 未證實 ⚠ |
| 第三方評價 | 「網球優先，pickleball 支援較弱」；分不出 drop 與 drive、不知旋轉 ⚠ |

## 10. 平台／定價（2026-09）

| 方案 | 價格 | 小時/月 | 內容 |
|---|---|---|---|
| Free | $0（廣告） | 8 | 去死時間；擊球統計與熱圖；HD；Watch 挑戰；語音線審；7 天保存；≤5 分鐘含 logo 匯出 |
| Plus ⚠ | $14.99/月 | ? | 僅 App Store 內購列出 |
| Pro | $14.99/月 或 $11.99/月年繳 + Swing Stick | 30 | + 逐拍分析與原始資料匯出；AI 計分與比賽統計；全場線審；無限分享與儲存 |
| Max | $39.99/月 或 $299.99/年；5 人家庭 | 60 | + Strategy Insights；4K；家庭共享；優先支援 |
| Teams | 洽談 | 60/人 | 團隊儀表板、直播、永久儲存、「AI Officiating」即將推出 |
| Swing Court | 年授權 | — | 硬體 + 軟體、ELC、kiosk、直播、排行榜 |

平台：iOS/iPadOS 18+、watchOS 10+、macOS 15+（匯入處理）、visionOS 2；無 Android app。運動：網球、pickleball；padel 即將推出。準確度：球速 ±10%、落點約 5% ⚠、近線 97%、整體 >99%。

## 無法證實的項目（介面設計時當 optional 欄位）

- Net clearance 高度、旋轉 RPM、跑動距離（公尺）、衝刺次數、姿態關鍵點
- Kitchen/NVZ 違例、腳誤、pickleball drive/lob/erne 分類、pickleball 旋轉
- Aces 作為獨立統計、`Direction` 的完整 enum、Points/Games/Sets 匯出欄位
- 雙打 AI Coaching 狀態、Plus 方案內容、Swing Court 雙鏡頭上市日期
