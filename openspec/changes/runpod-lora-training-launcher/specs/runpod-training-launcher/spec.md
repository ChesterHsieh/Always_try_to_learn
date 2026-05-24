---
project: lora-image-gen
---

# runpod-training-launcher Specification

## ADDED Requirements

### Requirement: 從設定檔載入並驗證啟動所需資訊

系統 SHALL 從一個 `.env` 檔載入所有啟動 RunPod 訓練所需的設定與機密（至少包含 RunPod API key、目標 Network Volume 識別碼、GPU 型別、rclone 對 Google Drive 的設定，以及訓練輸出在 Google Drive 的目的地路徑），並在啟動前驗證必要項是否齊全；缺少任一必要項時 SHALL 以清楚錯誤訊息與非零結束碼中止，不建立任何 pod。系統 SHALL 提供 `.env.example` 樣板列出所有設定鍵，且 SHALL NOT 將機密寫入版控或日誌。

#### Scenario: 必要設定齊全

- **WHEN** `.env` 含全部必要鍵且值非空
- **THEN** 系統載入設定、通過驗證，並繼續後續啟動流程

#### Scenario: 缺少必要設定

- **WHEN** `.env` 缺少任一必要鍵或其值為空
- **THEN** 系統 SHALL 列出缺少的鍵名、以非零結束碼中止，且不建立任何 pod 或上傳任何資料

#### Scenario: 機密不外洩

- **WHEN** 系統印出進度或錯誤訊息
- **THEN** 訊息 SHALL NOT 包含 API key、rclone token 等機密的明文內容

### Requirement: 一次性 pod 生命週期管理

系統 SHALL 使用 RunPod 官方 SDK，以既有的 Network Volume 建立帶 GPU 的 pod 來執行單次訓練；Network Volume SHALL 在建立 pod 時掛載，且其資料中心 SHALL 與 pod 一致。訓練流程結束後，系統 SHALL 依設定回收（terminate）該 pod 以停止計費。系統 SHALL NOT 嘗試建立 Network Volume（須由使用者預先建立並於設定中提供其識別碼）。

#### Scenario: 以既有 Volume 建立訓練 pod

- **WHEN** 設定提供有效的 Network Volume 識別碼與 GPU 型別
- **THEN** 系統建立掛載該 Volume 的 GPU pod，並回報 pod 識別碼與狀態

#### Scenario: 訓練結束後回收 pod

- **WHEN** 訓練流程結束（成功或失敗）且使用者未要求保留 pod
- **THEN** 系統 SHALL terminate 該 pod 並回報已回收

#### Scenario: 要求保留 pod 以供除錯

- **WHEN** 使用者帶入保留 pod 的旗標或設定
- **THEN** 系統 SHALL 在訓練結束後保留 pod 不回收，並提示使用者需自行回收以免持續計費

#### Scenario: 不建立 Network Volume

- **WHEN** 設定未提供 Network Volume 識別碼
- **THEN** 系統 SHALL 報錯中止，並指示使用者先於 RunPod 建立 Network Volume 再提供其識別碼

### Requirement: 上傳訓練資料集到 Pod

系統 SHALL 將本機準備好的訓練資料集（裁切圖與同名 caption `.txt`）送至 pod 的 Network Volume 指定資料夾，作為訓練輸入。當指定的本機資料集來源不存在、或不含任何成對的圖與 caption 時，系統 SHALL 報錯並以非零結束碼中止，不啟動訓練。

#### Scenario: 上傳有效資料集

- **WHEN** 本機資料集來源存在且含成對的圖與同名 `.txt`
- **THEN** 系統將整個資料集傳至 pod 上的訓練資料夾，並回報已上傳的圖片數

#### Scenario: 資料集來源不存在或為空

- **WHEN** 指定的本機資料集來源不存在，或不含任何成對的圖與 caption
- **THEN** 系統 SHALL 報錯、以非零結束碼中止，且不啟動訓練

### Requirement: 執行 LoRA 訓練

系統 SHALL 在 pod 上以訓練框架對上傳的資料集執行一次 SDXL LoRA 訓練，並把訓練好的 LoRA 權重與訓練 log 寫到 Network Volume 上的輸出位置。訓練超參數（如觸發詞、rank/alpha、學習率、步數）SHALL 可由設定帶入而不需改動程式碼。訓練以非零結束碼結束時，系統 SHALL 視為失敗並回報。

#### Scenario: 訓練成功產出 LoRA

- **WHEN** 資料集已上傳且訓練設定有效
- **THEN** 系統在 pod 上執行訓練，於 Network Volume 輸出 LoRA 權重檔與訓練 log，並回報輸出位置

#### Scenario: 以設定覆寫訓練超參數

- **WHEN** 使用者在設定中指定觸發詞或訓練超參數
- **THEN** 系統以該值執行訓練，無需修改程式碼

#### Scenario: 訓練失敗

- **WHEN** 訓練程序以非零結束碼結束
- **THEN** 系統 SHALL 將該次流程標記為失敗、回報失敗原因或 log 位置，並依設定決定是否回收 pod

### Requirement: 產出同步到 Google Drive

系統 SHALL 在 pod 上以 rclone 將訓練產出（至少含 LoRA 權重與訓練 log）同步到使用者 Google Drive 的設定目的地，rclone 的授權 SHALL 由設定注入而不在 pod 上進行互動式 OAuth。同步失敗時，系統 SHALL 回報失敗，且 SHALL NOT 在尚未確認產出已持久化前回收 pod。

#### Scenario: 同步產出到 Google Drive

- **WHEN** 訓練成功且 rclone 設定有效
- **THEN** 系統將 LoRA 權重與訓練 log 同步到設定指定的 Google Drive 路徑，並回報同步結果

#### Scenario: 同步失敗時保留產出

- **WHEN** rclone 同步以失敗結束
- **THEN** 系統 SHALL 回報同步失敗、SHALL NOT 在此情況下回收 pod，並提示產出仍在 Network Volume 上可手動取回

#### Scenario: 非互動式授權

- **WHEN** pod 上執行 rclone 同步
- **THEN** 系統 SHALL 使用由設定注入的既有授權（token），SHALL NOT 要求在 pod 上開瀏覽器進行互動授權

### Requirement: 除錯介面連線資訊

系統 SHALL 讓 pod 對外提供 ComfyUI Web UI（經 RunPod proxy 的 HTTP 服務）與 SSH 連線，並在啟動後印出兩者的連線資訊（ComfyUI 的 proxy URL 與 SSH 連線指令），供使用者於訓練期間或訓練後檢視與除錯。

#### Scenario: 印出除錯連線資訊

- **WHEN** pod 啟動完成且服務就緒
- **THEN** 系統 SHALL 印出 ComfyUI 的 proxy URL 與可直接使用的 SSH 連線指令

#### Scenario: 套用訓練好的 LoRA 驗證

- **WHEN** 使用者透過印出的 ComfyUI URL 連入，並載入訓練輸出的 LoRA
- **THEN** 使用者 SHALL 能在該 ComfyUI 介面以該 LoRA 產圖以肉眼驗證風格
