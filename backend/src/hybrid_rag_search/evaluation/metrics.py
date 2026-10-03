"""Document-level retrieval metrics with fixed cutoffs and graded relevance."""

import math
from collections.abc import Mapping, Sequence


def score_ranking(ranking: Sequence[str], judgments: Mapping[str, int]) -> dict[str, float]:
    """Deduplicate before cutoffs; unjudged results earn no credit."""
    if any(type(grade) is not int or not 0 <= grade <= 3 for grade in judgments.values()):
        raise ValueError("Relevance grades must be integers between zero and three")
    relevant = {key for key, grade in judgments.items() if grade > 0}
    if not relevant:
        raise ValueError("A query must have a positive judgment")
    documents = list(dict.fromkeys(ranking))
    scores = {
        f"recall@{cutoff}": len(set(documents[:cutoff]) & relevant) / len(relevant)
        for cutoff in (5, 10, 50)
    }
    scores["mrr@10"] = next(
        (1 / rank for rank, key in enumerate(documents[:10], 1) if key in relevant), 0.0
    )

    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))

    ideal = dcg(sorted(judgments.values(), reverse=True)[:10])
    scores["ndcg@10"] = dcg([judgments.get(key, 0) for key in documents[:10]]) / ideal
    return scores
