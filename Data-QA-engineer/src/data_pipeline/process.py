from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from .processor import clean_orders_dataframe, run_rules
from .utils.io import load_inputs as _load_inputs

IssueType = Literal[
    "invalid_product",
    "exceeded_inventory",
    "price_mismatch",
    "negative_quantity",
    "temporal_inconsistency",
    "missing_shipping_for_shipped",
    "potential_fraud_quantity",
]


@dataclass
class ProcessResult:
    clean_orders: pl.DataFrame
    issues: pl.DataFrame


# Backwards-compatible alias for callers importing from process.py
def load_inputs(
    orders_csv: Path, products_csv: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _load_inputs(orders_csv, products_csv)


def apply_business_rules(
    orders: pl.DataFrame, products: pl.DataFrame
) -> ProcessResult:
    """Apply all business rules to orders and products, return cleaned data and issues."""
    # Clone to avoid mutations
    orders = orders.clone()
    products = products.clone()

    # Ensure dates are parsed (Polars auto-detects but let's be explicit if needed)
    if orders.schema["order_date"] == pl.Utf8:
        orders = orders.with_columns(pl.col("order_date").str.to_date())
    if (
        "shipping_date" in orders.columns
        and orders.schema["shipping_date"] == pl.Utf8
    ):
        orders = orders.with_columns(pl.col("shipping_date").str.to_date())

    # Delegate all issue detection to rules module
    issues_df = run_rules(orders, products)

    # Clean orders via processor
    clean = clean_orders_dataframe(orders, products)

    return ProcessResult(clean_orders=clean, issues=issues_df)


def process_files(
    orders_csv: Path, products_csv: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Load CSV files, apply business rules, and write cleaned and issues CSVs."""
    orders, products = load_inputs(orders_csv, products_csv)
    result = apply_business_rules(orders, products)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "orders_clean.csv"
    issues_path = out_dir / "orders_issues.csv"
    result.clean_orders.write_csv(clean_path)
    result.issues.write_csv(issues_path)
    return clean_path, issues_path
