from __future__ import annotations

from pathlib import Path
from typing import Tuple

import polars as pl


def load_inputs(
    orders_csv: Path, products_csv: Path
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    orders = pl.read_csv(orders_csv)
    products = pl.read_csv(products_csv)
    return orders, products
