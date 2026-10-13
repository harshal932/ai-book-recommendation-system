"""Unit tests for RuleEngine and HybridRecommender."""

import pytest
import pandas as pd
from src.data_loader import load_books_dataset
from src.recommender import ContentBasedRecommender
from src.rule_engine import RuleEngine, UserPreferences, HybridRecommender


@pytest.fixture(scope="module")
def setup_hybrid():
    df = load_books_dataset()
    content_rec = ContentBasedRecommender(df)
    hybrid_rec = HybridRecommender(df, content_rec)
    return df, hybrid_rec


def test_rule_engine_empty_preferences():
    """Verify empty preferences produces a neutral baseline score and default rule explanation."""
    engine = RuleEngine()
    prefs = UserPreferences()
    book = {
        "genre": "Technology",
        "reading_level": "Intermediate",
        "rating": 4.5,
        "publication_year": 2018,
        "title": "Sample",
        "description": "Sample",
    }
    score, audit = engine.evaluate_book(book, prefs)
    assert 0.0 <= score <= 1.0
    assert len(audit) > 0
    assert "Default Rule" in audit[0]


def test_rule_engine_genre_match():
    """Verify a book matching the user's genre scores higher than a non-matching book."""
    engine = RuleEngine()
    prefs = UserPreferences(preferred_genres=["Technology"])

    tech_book = {
        "genre": "Technology",
        "reading_level": "Intermediate",
        "rating": 4.5,
        "publication_year": 2018,
    }
    fiction_book = {
        "genre": "Fiction",
        "reading_level": "Intermediate",
        "rating": 4.5,
        "publication_year": 2018,
    }

    score_tech, audit_tech = engine.evaluate_book(tech_book, prefs)
    score_fict, audit_fict = engine.evaluate_book(fiction_book, prefs)

    assert score_tech > score_fict
    assert any("Genre Match" in a for a in audit_tech)
    assert any("Genre Mismatch" in a for a in audit_fict)


def test_rule_engine_reading_level_matching():
    """Verify exact reading level match scores higher than mismatch."""
    engine = RuleEngine()
    prefs = UserPreferences(reading_level="Beginner")

    beg_book = {"genre": "Science", "reading_level": "Beginner", "rating": 4.0, "publication_year": 2015}
    adv_book = {"genre": "Science", "reading_level": "Advanced", "rating": 4.0, "publication_year": 2015}

    score_beg, _ = engine.evaluate_book(beg_book, prefs)
    score_adv, _ = engine.evaluate_book(adv_book, prefs)

    assert score_beg > score_adv


def test_rule_engine_rating_filter():
    """Verify rating threshold impacts rule audit and scoring."""
    engine = RuleEngine()
    prefs = UserPreferences(min_rating=4.7)

    high_rated = {"genre": "History", "reading_level": "Beginner", "rating": 4.8, "publication_year": 2015}
    low_rated = {"genre": "History", "reading_level": "Beginner", "rating": 4.2, "publication_year": 2015}

    score_high, audit_high = engine.evaluate_book(high_rated, prefs)
    score_low, audit_low = engine.evaluate_book(low_rated, prefs)

    assert score_high > score_low
    assert any("Rating Criterion Met" in a for a in audit_high)
    assert any("Rating Penalty" in a for a in audit_low)


def test_hybrid_recommender_scoring_and_normalization(setup_hybrid):
    """Verify hybrid recommendation scores are normalized between 0 and 1 and properly sorted."""
    df, hybrid = setup_hybrid
    prefs = UserPreferences(preferred_genres=["Technology"], reading_level="Intermediate")

    results = hybrid.recommend_hybrid(
        prefs=prefs,
        seed_book_id="B030",
        w_content=0.6,
        w_rule=0.4,
        top_n=5,
        exclude_seed=True,
    )

    assert len(results) == 5
    for r in results:
        assert 0.0 <= r["hybrid_score"] <= 1.0
        assert 0.0 <= r["content_similarity_score"] <= 1.0
        assert 0.0 <= r["rule_score"] <= 1.0
        assert "rules_fired" in r
        assert len(r["rules_fired"]) > 0

    # Ensure seed book is excluded
    assert "B030" not in [r["book_id"] for r in results]

    # Ensure sorted descending by hybrid score
    scores = [r["hybrid_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_weight_shift_behavior(setup_hybrid):
    """Verify that pure rule weight (w_content=0, w_rule=1) prioritizes rule match over content similarity."""
    df, hybrid = setup_hybrid
    prefs = UserPreferences(preferred_genres=["Philosophy"], reading_level="Beginner")

    results_rules_only = hybrid.recommend_hybrid(
        prefs=prefs,
        seed_book_id="B030",  # Clean Code (Technology)
        w_content=0.0,
        w_rule=1.0,
        top_n=3,
        exclude_seed=True,
    )

    assert len(results_rules_only) > 0
    # Top recommendation should match the requested rule genre (Philosophy)
    assert results_rules_only[0]["genre"] == "Philosophy"
