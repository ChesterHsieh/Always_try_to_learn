from __future__ import annotations

import polars as pl

from data_pipeline.rules import (
    RULES,
    rule_exceeded_inventory,
    rule_invalid_product,
    rule_negative_quantity,
    rule_potential_fraud_quantity,
    rule_price_mismatch,
    rule_temporal_inconsistency,
)


def _mk_orders(**overrides: object) -> pl.DataFrame:
    """Helper to create orders DataFrame with defaults."""
    base = {
        "order_id": [1],
        "customer_id": ["C1"],
        "product_id": ["P1"],
        "quantity": [1],
        "price": [10.0],
        "order_date": ["2025-01-01"],
        "shipping_date": ["2025-01-02"],
        "order_status": ["Shipped"],
    }
    base.update(overrides)
    df = pl.DataFrame(base)
    # Parse dates
    if "order_date" in df.columns:
        df = df.with_columns(pl.col("order_date").str.to_date())
    if "shipping_date" in df.columns:
        df = df.with_columns(pl.col("shipping_date").str.to_date())
    return df


def _mk_products(**overrides: object) -> pl.DataFrame:
    """Helper to create products DataFrame with defaults."""
    base = {
        "product_id": ["P1"],
        "product_name": ["A"],
        "category": ["G"],
        "inventory_count": [5],
        "price": [10.0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_rule_invalid_product_flags_unknown() -> None:
    """Test invalid_product rule flags unknown product IDs."""
    orders = _mk_orders(product_id=["P999"])  # not in products
    products = _mk_products()
    out = rule_invalid_product(orders, products)
    assert out.height > 0
    assert set(out["issue"].to_list()) == {"invalid_product"}


def test_rule_invalid_product_ok_when_known() -> None:
    """Test invalid_product rule passes for known product IDs."""
    orders = _mk_orders(product_id=["P1"])  # exists in products
    products = _mk_products()
    out = rule_invalid_product(orders, products)
    assert out.height == 0


def test_rule_exceeded_inventory_flags_when_quantity_gt_inventory() -> None:
    """Test exceeded_inventory rule flags when quantity exceeds stock."""
    orders = _mk_orders(quantity=[10])
    products = _mk_products(inventory_count=[5])
    out = rule_exceeded_inventory(orders, products)
    assert out.height > 0
    assert set(out["issue"].to_list()) == {"exceeded_inventory"}


def test_rule_exceeded_inventory_ok_when_within_inventory() -> None:
    """Test exceeded_inventory rule passes when within stock."""
    orders = _mk_orders(quantity=[3])
    products = _mk_products(inventory_count=[5])
    out = rule_exceeded_inventory(orders, products)
    assert out.height == 0


def test_rule_price_mismatch_flags_when_price_differs() -> None:
    """Test price_mismatch rule flags price discrepancies."""
    orders = _mk_orders(price=[12.0])
    products = _mk_products(price=[10.0])
    out = rule_price_mismatch(orders, products)
    assert out.height > 0
    assert set(out["issue"].to_list()) == {"price_mismatch"}


def test_rule_price_mismatch_ok_when_matches() -> None:
    """Test price_mismatch rule passes when prices match."""
    orders = _mk_orders(price=[10.0])
    products = _mk_products(price=[10.0])
    out = rule_price_mismatch(orders, products)
    assert out.height == 0


def test_rule_negative_quantity_flags_negative() -> None:
    """Test negative_quantity rule flags negative quantities."""
    orders = _mk_orders(quantity=[-1])
    products = _mk_products()
    out = rule_negative_quantity(orders, products)
    assert out.height > 0
    assert set(out["issue"].to_list()) == {"negative_quantity"}


def test_rule_negative_quantity_ok_when_non_negative() -> None:
    """Test negative_quantity rule passes for zero/positive quantities."""
    orders = _mk_orders(quantity=[0])
    products = _mk_products()
    out = rule_negative_quantity(orders, products)
    assert out.height == 0


def test_rule_temporal_inconsistency_flags_before_order() -> None:
    """Test temporal_inconsistency flags shipping before order date."""
    orders = _mk_orders(
        shipping_date=["2024-12-31"],
        order_date=["2025-01-01"],
        order_status=["Shipped"],
    )
    products = _mk_products()
    out = rule_temporal_inconsistency(orders, products)
    assert out.height > 0
    assert "temporal_inconsistency" in set(out["issue"].to_list())


def test_rule_temporal_inconsistency_flags_missing_shipping_for_shipped() -> (
    None
):
    """Test temporal_inconsistency flags missing shipping date for shipped orders."""
    orders = pl.DataFrame(
        {
            "order_id": [1],
            "customer_id": ["C1"],
            "product_id": ["P1"],
            "quantity": [1],
            "price": [10.0],
            "order_date": ["2025-01-01"],
            "shipping_date": [None],
            "order_status": ["Shipped"],
        }
    ).with_columns(pl.col("order_date").str.to_date())
    products = _mk_products()
    out = rule_temporal_inconsistency(orders, products)
    assert out.height > 0
    assert "missing_shipping_for_shipped" in set(out["issue"].to_list())


def test_rule_temporal_inconsistency_ok_when_pending_and_missing_shipping() -> (
    None
):
    """Test temporal_inconsistency passes when pending and shipping missing."""
    orders = pl.DataFrame(
        {
            "order_id": [1],
            "customer_id": ["C1"],
            "product_id": ["P1"],
            "quantity": [1],
            "price": [10.0],
            "order_date": ["2025-01-01"],
            "shipping_date": [None],
            "order_status": ["Pending"],
        }
    ).with_columns(pl.col("order_date").str.to_date())
    products = _mk_products()
    out = rule_temporal_inconsistency(orders, products)
    assert out.height == 0


def test_rule_potential_fraud_quantity_flags_large_spike() -> None:
    """Test potential_fraud_quantity flags large spikes beyond 2 std from mean."""
    # Build 6 entries for same product so rolling window has >=5 history rows
    dates = [f"2025-01-{d:02d}" for d in [1, 2, 3, 4, 5, 6]]
    orders = pl.DataFrame(
        {
            "order_id": list(range(1, 7)),
            "customer_id": ["C1"] * 6,
            "product_id": ["P1"] * 6,
            "quantity": [3, 3, 4, 3, 3, 20],  # last should be outlier
            "price": [10.0] * 6,
            "order_date": dates,
            "shipping_date": dates,
            "order_status": ["Shipped"] * 6,
        }
    ).with_columns(
        [
            pl.col("order_date").str.to_date(),
            pl.col("shipping_date").str.to_date(),
        ]
    )
    products = _mk_products()
    out = rule_potential_fraud_quantity(orders, products)
    assert out.height > 0
    assert set(out["issue"].to_list()) == {"potential_fraud_quantity"}


def test_rule_potential_fraud_quantity_ok_when_normal_range() -> None:
    """Test potential_fraud_quantity passes when all values near mean."""
    dates = [f"2025-01-{d:02d}" for d in [1, 2, 3, 4, 5, 6]]
    orders = pl.DataFrame(
        {
            "order_id": list(range(1, 7)),
            "customer_id": ["C1"] * 6,
            "product_id": ["P1"] * 6,
            "quantity": [3, 4, 3, 4, 3, 4],  # all near mean, below threshold
            "price": [10.0] * 6,
            "order_date": dates,
            "shipping_date": dates,
            "order_status": ["Shipped"] * 6,
        }
    ).with_columns(
        [
            pl.col("order_date").str.to_date(),
            pl.col("shipping_date").str.to_date(),
        ]
    )
    products = _mk_products()
    out = rule_potential_fraud_quantity(orders, products)
    assert out.height == 0


def test_rules_registry_contains_all() -> None:
    """Smoke test: verify RULES registry contains all expected rules."""
    names = {fn.__name__ for fn in RULES}
    assert {
        "rule_invalid_product",
        "rule_exceeded_inventory",
        "rule_price_mismatch",
        "rule_negative_quantity",
        "rule_temporal_inconsistency",
        "rule_potential_fraud_quantity",
    }.issubset(names)
