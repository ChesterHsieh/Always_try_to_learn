from __future__ import annotations

import polars as pl


def _columns_for_issue(df: pl.DataFrame) -> list[str]:
    """Return subset of columns for issue reporting."""
    cols = [
        "order_id",
        "product_id",
        "quantity",
        "price",
        "order_date",
        "shipping_date",
        "order_status",
    ]
    return [c for c in cols if c in df.columns]


def rule_invalid_product(
    orders: pl.DataFrame, products: pl.DataFrame
) -> pl.DataFrame:
    """Flag orders with product_id not in products inventory."""
    valid = products.select("product_id").to_series().cast(str).to_list()
    result = (
        orders.filter(~pl.col("product_id").cast(str).is_in(valid))
        .select(_columns_for_issue(orders))
        .with_columns(pl.lit("invalid_product").alias("issue"))
    )
    return result


def rule_exceeded_inventory(
    orders: pl.DataFrame, products: pl.DataFrame
) -> pl.DataFrame:
    """Flag orders where quantity exceeds available inventory."""
    result = (
        orders.join(
            products.select(["product_id", "inventory_count"]),
            on="product_id",
            how="left",
        )
        .filter(pl.col("quantity") > pl.col("inventory_count").fill_null(-1))
        .select(_columns_for_issue(orders))
        .with_columns(pl.lit("exceeded_inventory").alias("issue"))
    )
    return result


def rule_price_mismatch(
    orders: pl.DataFrame, products: pl.DataFrame
) -> pl.DataFrame:
    """Flag orders where price doesn't match product inventory price."""
    result = (
        orders.join(
            products.select(
                ["product_id", pl.col("price").alias("expected_price")]
            ),
            on="product_id",
            how="left",
        )
        .filter(pl.col("price").round(2) != pl.col("expected_price").round(2))
        .select(_columns_for_issue(orders))
        .with_columns(pl.lit("price_mismatch").alias("issue"))
    )
    return result


def rule_negative_quantity(
    orders: pl.DataFrame,
    products: pl.DataFrame,  # noqa: ARG001
) -> pl.DataFrame:
    """Flag orders with negative quantity."""
    result = (
        orders.filter(pl.col("quantity") < 0)
        .select(_columns_for_issue(orders))
        .with_columns(pl.lit("negative_quantity").alias("issue"))
    )
    return result


def rule_temporal_inconsistency(
    orders: pl.DataFrame,
    products: pl.DataFrame,  # noqa: ARG001
) -> pl.DataFrame:
    """Flag temporal issues: shipping before order or missing shipping for shipped orders."""
    # Cast dates if they're strings
    df = orders.clone()
    if df.schema["order_date"] == pl.Utf8:
        df = df.with_columns(pl.col("order_date").str.to_date())
    if df.schema["shipping_date"] == pl.Utf8:
        df = df.with_columns(pl.col("shipping_date").str.to_date())

    # Shipping before order
    before = (
        df.filter(
            pl.col("shipping_date").is_not_null()
            & (pl.col("shipping_date") < pl.col("order_date"))
        )
        .select(_columns_for_issue(df))
        .with_columns(pl.lit("temporal_inconsistency").alias("issue"))
    )

    # Missing shipping for shipped orders
    missing = (
        df.filter(
            (pl.col("order_status").str.to_lowercase() == "shipped")
            & pl.col("shipping_date").is_null()
        )
        .select(_columns_for_issue(df))
        .with_columns(pl.lit("missing_shipping_for_shipped").alias("issue"))
    )

    # Combine results
    if before.height > 0 and missing.height > 0:
        return pl.concat([before, missing])
    elif before.height > 0:
        return before
    elif missing.height > 0:
        return missing
    else:
        cols = _columns_for_issue(df) + ["issue"]
        return pl.DataFrame(
            {c: [] for c in cols},
            schema={c: df.schema.get(c, pl.Utf8) for c in cols[:-1]}
            | {"issue": pl.Utf8},
        )


def rule_potential_fraud_quantity(
    orders: pl.DataFrame,
    products: pl.DataFrame,  # noqa: ARG001
) -> pl.DataFrame:
    """Flag orders with quantity significantly above historical patterns (>2 std from mean in last 30 days)."""
    # Ensure order_date is date type
    df = orders.clone()
    if df.schema["order_date"] == pl.Utf8:
        df = df.with_columns(pl.col("order_date").str.to_date())

    # Sort by product and date
    df = df.sort(["product_id", "order_date"])

    # Use window functions for rolling statistics per product
    # For each row, compute mean and std of previous 30 days with at least 5 samples
    result = (
        df.with_row_index("__idx")
        .with_columns(
            [
                # Rolling window: previous rows within 30 days
                pl.col("quantity")
                .rolling_mean_by(
                    by="order_date",
                    window_size="30d",
                    closed="left",
                )
                .over("product_id")
                .alias("mean_30d"),
                pl.col("quantity")
                .rolling_std_by(
                    by="order_date",
                    window_size="30d",
                    closed="left",
                )
                .over("product_id")
                .alias("std_30d"),
                # Count samples in window using cumcount as proxy
                pl.col("quantity")
                .cum_count()
                .over("product_id")
                .alias("count_30d"),
            ]
        )
        .filter(
            # Need at least 5 historical samples
            (pl.col("count_30d") > 5)  # >5 means at least 5 prior + current
            # Quantity exceeds mean + 2*std
            & (pl.col("mean_30d").is_not_null())
            & (
                pl.col("quantity")
                > pl.col("mean_30d") + 2 * pl.col("std_30d").fill_null(1.0)
            )
        )
        .select(_columns_for_issue(orders))
        .with_columns(pl.lit("potential_fraud_quantity").alias("issue"))
    )

    return result


RULES = [
    rule_invalid_product,
    rule_exceeded_inventory,
    rule_price_mismatch,
    rule_negative_quantity,
    rule_temporal_inconsistency,
    rule_potential_fraud_quantity,
]
