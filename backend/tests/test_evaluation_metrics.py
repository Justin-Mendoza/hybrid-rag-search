import math

import pytest

from hybrid_rag_search.evaluation.metrics import score_ranking


def test_known_answer_graded_ranking() -> None:
    scores = score_ranking(["negative", "weak", "best"], {"weak": 1, "best": 3, "negative": 0})
    assert scores["recall@5"] == scores["recall@10"] == scores["recall@50"] == 1
    assert scores["mrr@10"] == 0.5
    assert scores["ndcg@10"] == pytest.approx((1 / math.log2(3) + 7 / 2) / (7 + 1 / math.log2(3)))


def test_duplicates_are_removed_before_cutoffs_and_short_results_keep_denominator() -> None:
    scores = score_ranking(["a"] * 60 + ["b"], {"a": 3, "b": 2, "c": 1})
    assert scores["recall@50"] == scores["recall@5"] == 2 / 3
    assert scores["mrr@10"] == 1


def test_cutoffs_and_unjudged_results() -> None:
    scores = score_ranking([f"unknown-{n}" for n in range(10)] + ["a"], {"a": 3})
    assert scores == {"recall@5": 0, "recall@10": 0, "recall@50": 1, "mrr@10": 0, "ndcg@10": 0}
    assert score_ranking([], {"a": 1}) == dict.fromkeys(scores, 0)


@pytest.mark.parametrize("grades", [{"a": 0}, {"a": -1}, {"a": 4}, {"a": True}])
def test_invalid_judgments(grades: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        score_ranking(["a"], grades)
