from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.dates import random_dates

ORDERS_SCHEMA = [
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "price",
    "order_date",
    "shipping_date",
    "order_status",
]

PRODUCTS_SCHEMA = [
    "product_id",
    "product_name",
    "category",
    "inventory_count",
    "price",
]


@dataclass
class SampleDataPaths:
    base_dir: Path

    @property
    def orders_csv(self) -> Path:
        return self.base_dir / "orders.csv"

    @property
    def products_csv(self) -> Path:
        return self.base_dir / "product_inventory.csv"


def create_sample_csvs(paths: SampleDataPaths) -> None:
    paths.base_dir.mkdir(parents=True, exist_ok=True)

    # Products table from the image
    products = pd.DataFrame(
        [
            ["P001", "Widget A", "Gadgets", 100, 100.00],
            ["P002", "Widget B", "Gadgets", 50, 120.00],
            ["P003", "Widget C", "Tools", 30, 50.00],
            ["P004", "Widget D", "Tools", 200, 80.00],
            ["P005", "Widget E", "Furniture", 0, 30.00],
            ["P006", "Widget F", "Furniture", 10, 200.00],
        ],
        columns=PRODUCTS_SCHEMA,
    )
    products.to_csv(paths.products_csv, index=False)

    # Orders table from the image
    orders = pd.DataFrame(
        [
            [
                1001,
                "C001",
                "P001",
                2,
                100.00,
                "2025-05-01",
                "2025-05-03",
                "Shipped",
            ],
            [
                1002,
                "C002",
                "P002",
                5,
                120.00,
                "2025-05-02",
                "2025-05-06",
                "Pending",
            ],
            [
                1003,
                "C003",
                "P005",
                1,
                30.00,
                "2025-05-03",
                "2025-05-04",
                "Cancelled",
            ],
            [
                1004,
                "C004",
                "P001",
                3,
                100.00,
                "2025-05-04",
                "2025-05-07",
                "Shipped",
            ],
            [1005, "C005", "P003", 10, 50.00, "2025-05-04", None, "Pending"],
            [
                1006,
                "C006",
                "P006",
                1,
                200.00,
                "2025-05-05",
                "2025-05-06",
                "Shipped",
            ],
        ],
        columns=ORDERS_SCHEMA,
    )
    orders.to_csv(paths.orders_csv, index=False)


def generate_random_data(
    out_dir: Path,
    num_products: int = 50,
    num_orders: int = 500,
    seed: int | None = 42,
) -> tuple[Path, Path]:
    if seed is not None:
        np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Products
    categories = ["Gadgets", "Tools", "Furniture", "Home", "Sports"]
    product_ids = [f"P{i:04d}" for i in range(1, num_products + 1)]
    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": [
                f"Product {i:04d}" for i in range(1, num_products + 1)
            ],
            "category": np.random.choice(
                categories, size=num_products
            ).tolist(),
            "inventory_count": np.random.randint(0, 500, size=num_products),
            "price": np.round(np.random.uniform(5, 500, size=num_products), 2),
        }
    )
    products_path = out_dir / "product_inventory.csv"
    products.to_csv(products_path, index=False)

    # Orders
    order_ids = np.arange(1, num_orders + 1) + 1000
    customers = [
        f"C{i:04d}" for i in np.random.randint(1, 1000, size=num_orders)
    ]
    product_choices = np.random.choice(product_ids, size=num_orders)
    base_date = date.today() - timedelta(days=120)
    order_dates = random_dates(
        base_date, base_date + timedelta(days=120), num_orders
    )

    quantities = np.clip(
        np.random.normal(loc=3.0, scale=2.0, size=num_orders).astype(int),
        1,
        50,
    )
    statuses = np.random.choice(
        ["Pending", "Shipped", "Cancelled"], p=[0.3, 0.6, 0.1], size=num_orders
    )

    # Shipping date sometimes missing or before order (to test checks)
    shipping_dates: list[date | None] = []
    for i in range(num_orders):
        if statuses[i] == "Shipped" and np.random.rand() > 0.05:
            ship_delay = np.random.randint(0, 10)
            shipping_dates.append(order_dates[i] + timedelta(days=ship_delay))
        elif statuses[i] == "Cancelled":
            shipping_dates.append(None)
        else:
            # sometimes missing even if pending
            shipping_dates.append(
                None
                if np.random.rand() > 0.5
                else order_dates[i] + timedelta(days=2)
            )

    # Join prices from products
    price_map = {
        row.product_id: row.price for row in products.itertuples(index=False)
    }
    prices = [float(price_map[p]) for p in product_choices]

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customers,
            "product_id": product_choices,
            "quantity": quantities,
            "price": prices,
            "order_date": [d.isoformat() for d in order_dates],
            "shipping_date": [
                d.isoformat() if d else None for d in shipping_dates
            ],
            "order_status": statuses,
        }
    )

    orders_path = out_dir / "orders.csv"
    orders.to_csv(orders_path, index=False)
    return orders_path, products_path
