"""Unit tests for dataset loading, sanitization, and summary functions."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import (
    load_books_dataset,
    clean_books_dataframe,
    get_dataset_statistics,
    REQUIRED_COLUMNS,
)


def test_load_books_dataset_success():
    """Verify that default books dataset loads correctly and contains 100+ items."""
    df = load_books_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_clean_books_dataframe_deduplication():
    """Verify duplicate book_id values are removed."""
    raw_data = pd.DataFrame([
        {
            "book_id": "B001",
            "title": "Book One",
            "author": "Author A",
            "genre": "Fiction",
            "description": "Desc A",
            "publication_year": 2020,
            "rating": 4.5,
            "reading_level": "Beginner",
        },
        {
            "book_id": "B001",  # duplicate ID
            "title": "Book Duplicate",
            "author": "Author B",
            "genre": "Fiction",
            "description": "Desc B",
            "publication_year": 2021,
            "rating": 4.0,
            "reading_level": "Intermediate",
        },
    ])
    cleaned = clean_books_dataframe(raw_data)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["title"] == "Book One"


def test_clean_books_dataframe_missing_values():
    """Verify that missing fields (ratings, reading levels, years) are gracefully handled."""
    raw_data = pd.DataFrame([
        {
            "book_id": "B999",
            "title": "",
            "author": "",
            "genre": "",
            "description": None,
            "publication_year": "invalid_year",
            "rating": 9.9,  # out of bounds, should clip to 5.0
            "reading_level": "UnrecognizedLevel",
        }
    ])
    cleaned = clean_books_dataframe(raw_data)
    assert cleaned.iloc[0]["title"] == "Untitled Book (B999)"
    assert cleaned.iloc[0]["genre"] == "General"
    assert cleaned.iloc[0]["reading_level"] == "Intermediate"
    assert cleaned.iloc[0]["publication_year"] == 2000
    assert cleaned.iloc[0]["rating"] == 5.0  # clipped


def test_missing_csv_file_raises_error(tmp_path):
    """Verify clear FileNotFoundError when path does not exist."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_books_dataset(csv_path=fake_path)


def test_dataset_statistics():
    """Verify correct calculation of dataset summary metrics."""
    df = load_books_dataset()
    stats = get_dataset_statistics(df)
    assert stats["total_books"] == len(df)
    assert stats["unique_genres"] >= 5
    assert 1.0 <= stats["avg_rating"] <= 5.0
    assert "Beginner" in stats["reading_levels"]
