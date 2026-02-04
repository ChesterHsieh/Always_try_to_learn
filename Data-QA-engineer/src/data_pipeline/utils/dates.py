from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def random_dates(start: date, end: date, n: int, seed: int | None = None) -> list[date]:
    if seed is not None:
        np.random.seed(seed)
    delta = (end - start).days
    return [start + timedelta(days=int(x)) for x in np.random.randint(0, max(delta, 1), size=n)]
