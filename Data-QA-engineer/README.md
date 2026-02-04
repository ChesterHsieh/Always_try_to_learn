# Data Pipeline (uv)

A minimal, production-ready template for a data pipeline managed by `uv`.

- Dependency manager: `uv`
- Packaging: `pyproject.toml` with `src/` layout
- CLI: `typer`
- Data processing: `polars` (high-performance DataFrame library)
- Orchestration: `prefect`
- Config: `.env` and YAML
- Tooling: `ruff`, `mypy`, `pytest`

## Quickstart

1) Install uv (macOS/Linux):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create and activate virtualenv in project (uv manages it):

```
uv venv .venv
```

3) Sync dependencies (prod + dev):

```
uv sync --dev
```

4) Verify CLI and run example commands:

```
uv run pipeline --help
uv run pipeline version
uv run pipeline generate-sample data
uv run pipeline process data/orders.csv data/product_inventory.csv outputs

```

5) Run tests and linters:

```
uv run pytest -q
uv run ruff check .
uv run mypy src
```

## Developing Guideline

常見擴充點與對應檔案位置（依需求快速定位修改）：

- 規則擴充（新增/調整商業規則）: `src/data_pipeline/rules.py`
  - 新增一個 `rule_xxx(orders, products) -> pd.DataFrame` 並加入 `RULES` 清單
  - 規則應輸出原欄位 + `issue` 欄指明原因
- 清理邏輯擴充（清理與欄位標準化）: `src/data_pipeline/processor.py`
  - 編輯 `clean_orders_dataframe` 或新增函式供 `process.py` 呼叫
- 流程編排（輸入/規則/輸出組合）: `src/data_pipeline/process.py`
  - 調整 `apply_business_rules` 與 `process_files` 的串接與輸出
- 資料模型（欄位定義與驗證）: `src/data_pipeline/models.py`
  - 使用 Pydantic 定義 `Order`/`Product` 欄位條件
- 設定與組態（.env / YAML 載入）: `src/data_pipeline/utils/settings.py`
  - 擴充 `AppSettings` 欄位或變更 `load_config()` 行為
- IO 與工具（載入/存檔/日期工具）:
  - 讀寫 CSV: `src/data_pipeline/utils/io.py`
  - 日期/轉型: `src/data_pipeline/utils/dates.py`
- CLI 指令（新增子命令或參數）: `src/data_pipeline/utils/cli.py`
  - 目前指令：`version`、`generate-sample`、`process`
- 範例/隨機資料產生器: `src/data_pipeline/utils/generator.py`
  - 更新樣本資料 schema 或隨機資料參數
- 日誌初始化: `src/data_pipeline/logging_setup.py`
  - 調整日誌格式/等級/輸出目標

小技巧：
- 若新增檔案，遵循 `src/` 佈局並以套件路徑匯入（例如 `data_pipeline.utils.*`）。
- 新規則影響測試時，可在 `tests/` 內新增對應案例（參考 `tests/test_processing.py`）。


## Configuration

- Copy `.env.example` to `.env` and adjust values as needed. Environment variables with prefix `DP_` are loaded automatically.
- Default YAML config at `configs/config.yaml`.

## Notes

- **Polars vs Pandas**: This project uses Polars for data processing, offering significant performance improvements over Pandas for larger datasets. All DataFrames are `polars.DataFrame` instances.
- Pydantic v2 moves `BaseSettings` to `pydantic-settings` (already configured).
- If you encounter issues with `typer[all]`, note that this template uses plain `typer`.
- This project uses a `src/` layout with `pythonpath = ["src"]` configured in `pyproject.toml` for pytest.
