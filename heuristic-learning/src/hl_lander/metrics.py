"""Pure aggregation over EpisodeResult lists. No IO."""

from __future__ import annotations

import statistics
from typing import Sequence, TypedDict

from .runner import EpisodeResult


class Summary(TypedDict):
    n: int
    mean_return: float
    std_return: float
    landing_rate: float


def summarize(results: Sequence[EpisodeResult]) -> Summary:
    if not results:
        return Summary(n=0, mean_return=0.0, std_return=0.0, landing_rate=0.0)
    returns = [r.total_return for r in results]
    n = len(results)
    return Summary(
        n=n,
        mean_return=statistics.fmean(returns),
        std_return=statistics.pstdev(returns) if n > 1 else 0.0,
        landing_rate=sum(1 for r in results if r.landed) / n,
    )
