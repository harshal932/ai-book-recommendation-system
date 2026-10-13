"""Unit tests for Content-Based Recommendation Engine."""

import pytest
import pandas as pd
from src.data_loader import load_books_dataset
from src.recommender import ContentBasedRecommender


@pytest.fixture(scope="module")
def recommender():
    """Load default dataset and construct ContentBasedRecommender."""
    df = load_books_dataset()
    return ContentBasedRecommender(df)


def test_recommender_initialization_with_empty_df():
    """Verify error on empty DataFrame initialization."""
    with pytest.raises(ValueError):
        ContentBasedRecommender(pd.DataFrame())


def test_recommend_similar_books_excludes_self(recommender):
    """Verify that the selected seed book is excluded from recommendations."""
    target_id = "B030"  # Clean Code
    results = recommender.get_similar_books(book_id=target_id, top_n=5, exclude_self=True)

    assert len(results) > 0
    assert len(results) <= 5

    recommended_ids = [r["book_id"] for r in results]
    assert target_id not in recommended_ids, "Target book must not recommend itself."


def test_recommend_similar_books_similarity_scores(recommender):
    """Verify similarity scores are bounded between 0.0 and 1.0 and sorted descending."""
    results = recommender.get_similar_books(book_id="B034", top_n=5)  # AI: A Modern Approach
    assert len(results) > 0

    scores = [r["similarity_score"] for r in results]
    for s in scores:
        assert 0.0 <= s <= 1.0

    # Ensure sorted descending
    assert scores == sorted(scores, reverse=True)


def test_recommend_unknown_book_id(recommender):
    """Verify unknown book IDs return an empty list gracefully."""
    results = recommender.get_similar_books(book_id="NON_EXISTENT_ID_999", top_n=5)
    assert results == []


def test_recommend_by_query(recommender):
    """Verify free-text search returns relevant results."""
    query = "artificial intelligence machine learning deep algorithms"
    results = recommender.recommend_by_query(query_text=query, top_n=5)

    assert len(results) > 0
    assert len(results) <= 5
    for r in results:
        assert "book_id" in r
        assert "similarity_score" in r
        assert r["similarity_score"] > 0.0


def test_recommend_by_empty_query(recommender):
    """Verify empty or whitespace query returns empty list without error."""
    assert recommender.recommend_by_query("") == []
    assert recommender.recommend_by_query("   ") == []


def test_explainability_common_terms(recommender):
    """Verify explainability metadata is provided."""
    results = recommender.get_similar_books(book_id="B030", top_n=3)
    for r in results:
        assert "explanation" in r
        assert len(r["explanation"]) > 0
        assert "common_terms" in r
