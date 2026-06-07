"""Pure functions for evaluating top-K recommendations."""

from collections.abc import Iterable
import math
from typing import Hashable


ItemId = Hashable


def _top_k(recommended_ids: Iterable[ItemId], k: int) -> list[ItemId]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return list(recommended_ids)[:k]


def precision_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Return the fraction of the first K recommendations that are relevant."""
    recommendations = _top_k(recommended_ids, k)
    relevant = set(relevant_ids)
    hits = sum(item_id in relevant for item_id in recommendations)

    return hits / k


def recall_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Return the fraction of relevant items retrieved in the first K positions."""
    recommendations = _top_k(recommended_ids, k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = len(set(recommendations).intersection(relevant))
    return hits / len(relevant)


def dcg_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Return binary Discounted Cumulative Gain for the first K positions."""
    recommendations = _top_k(recommended_ids, k)
    relevant = set(relevant_ids)

    return sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(recommendations)
        if item_id in relevant
    )


def ndcg_at_k(
    recommended_ids: Iterable[ItemId],
    relevant_ids: Iterable[ItemId],
    k: int,
) -> float:
    """Return normalized binary DCG for the first K positions."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    return dcg_at_k(recommended_ids, relevant, k) / ideal_dcg


def catalog_coverage(
    all_recommended_ids: Iterable[ItemId],
    total_items: int,
) -> float:
    """Return the share of catalog items recommended at least once."""
    if total_items <= 0:
        raise ValueError("total_items must be greater than zero.")

    return min(len(set(all_recommended_ids)) / total_items, 1.0)
