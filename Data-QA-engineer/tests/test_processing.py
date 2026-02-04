from __future__ import annotations

from pathlib import Path

from data_pipeline.process import apply_business_rules, load_inputs
from data_pipeline.utils.generator import SampleDataPaths, create_sample_csvs


def test_sample_processing(tmp_path: Path) -> None:
    """Integration test: generate sample data, load, and verify processing results."""
    data_dir = tmp_path / "data"
    paths = SampleDataPaths(base_dir=data_dir)
    create_sample_csvs(paths)

    orders, products = load_inputs(paths.orders_csv, paths.products_csv)
    result = apply_business_rules(orders, products)

    # Clean set should be non-empty and keep columns
    assert result.clean_orders.height > 0
    for col in [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "price",
        "order_date",
        "shipping_date",
        "order_status",
    ]:
        assert col in result.clean_orders.columns

    # Sample should at least flag exceeded inventory (P005 has inventory 0)
    issues = set(result.issues["issue"].to_list())
    assert "exceeded_inventory" in issues
