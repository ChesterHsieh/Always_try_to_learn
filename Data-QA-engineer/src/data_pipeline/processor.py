from __future__ import annotations

import polars as pl

from .rules import RULES


def run_rules(orders: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    """Run all validation rules and return concatenated issues DataFrame."""
    frames = [rule(orders, products) for rule in RULES]
    frames = [f for f in frames if f.height > 0]
    return pl.concat(frames) if frames else pl.DataFrame()


def clean_orders_dataframe(
    orders: pl.DataFrame, products: pl.DataFrame
) -> pl.DataFrame:
    """Clean orders by removing invalid products, negative quantities, and clipping to inventory."""
    # Get valid product IDs
    valid_products = (
        products.select("product_id").to_series().cast(str).to_list()
    )

    # Join with inventory and filter/clip
    result = (
        orders.join(
            products.select(["product_id", "inventory_count"]),
            on="product_id",
            how="left",
        )
        # Filter: valid products and non-negative quantity
        .filter(
            pl.col("product_id").cast(str).is_in(valid_products)
            & (pl.col("quantity") >= 0)
        )
        # Clip quantity to inventory
        .with_columns(
            pl.when(pl.col("quantity") > pl.col("inventory_count"))
            .then(pl.col("inventory_count"))
            .otherwise(pl.col("quantity"))
            .alias("quantity")
        )
        # Drop helper column
        .drop("inventory_count")
    )
    return result
