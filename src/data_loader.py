"""Data loading, validation, and preprocessing module for the AI Book Recommendation System.

Handles:
- Loading CSV dataset from configurable paths.
- Schema verification and missing field validation.
- Handling missing descriptions, ratings, and invalid values gracefully.
- Normalizing reading levels, genre casing, and numeric types.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "book_id",
    "title",
    "author",
    "genre",
    "description",
    "publication_year",
    "rating",
    "reading_level",
]

VALID_READING_LEVELS = {"Beginner", "Intermediate", "Advanced"}


def get_default_data_path() -> Path:
    """Returns the default absolute path to data/books.csv relative to this module."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "books.csv"


def load_books_dataset(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load, validate, and clean the books dataset from a CSV file.

    Args:
        csv_path: Optional path to the CSV file. If None, uses default project data/books.csv.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame of books.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    if csv_path is None:
        target_path = get_default_data_path()
    else:
        target_path = Path(csv_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Books dataset file not found at: {target_path}. "
            "Please ensure data/books.csv exists or run initialization."
        )

    # Read CSV safely with explicit encoding
    try:
        df = pd.read_csv(target_path, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV at {target_path}: {exc}") from exc

    # Check for required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    df = clean_books_dataframe(df)
    return df


def clean_books_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, type-cast, and sanitize a raw books DataFrame.

    Args:
        df: Raw DataFrame containing books data.

    Returns:
        Cleaned copy of the DataFrame.
    """
    df = df.copy()

    # Drop duplicate IDs, keeping first occurrence
    df.drop_duplicates(subset=["book_id"], keep="first", inplace=True)

    # Clean textual fields
    text_cols = ["book_id", "title", "author", "genre", "description"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Ensure title is never empty (fallback to book_id if title missing)
    mask_empty_title = df["title"] == ""
    df.loc[mask_empty_title, "title"] = "Untitled Book (" + df.loc[mask_empty_title, "book_id"] + ")"

    # Clean genre
    df.loc[df["genre"] == "", "genre"] = "General"
    df["genre"] = df["genre"].str.title()

    # Clean and validate reading_level
    df["reading_level"] = df["reading_level"].fillna("Intermediate").astype(str).str.strip().str.capitalize()
    df["reading_level"] = df["reading_level"].apply(
        lambda x: x if x in VALID_READING_LEVELS else "Intermediate"
    )

    # Coerce publication_year to numeric, with fallback to 2000
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").fillna(2000).astype(int)

    # Coerce rating to float in range [1.0, 5.0], fallback to 4.0 if missing/invalid
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
    df["rating"] = df["rating"].clip(lower=1.0, upper=5.0).round(2)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key summary metrics for the loaded dataset.

    Args:
        df: Cleaned books DataFrame.

    Returns:
        Dictionary with count, unique authors, unique genres, rating stats, etc.
    """
    if df.empty:
        return {
            "total_books": 0,
            "unique_genres": 0,
            "unique_authors": 0,
            "genres_list": [],
            "avg_rating": 0.0,
            "min_year": 0,
            "max_year": 0,
        }

    return {
        "total_books": int(len(df)),
        "unique_genres": int(df["genre"].nunique()),
        "unique_authors": int(df["author"].nunique()),
        "genres_list": sorted(df["genre"].unique().tolist()),
        "avg_rating": float(round(df["rating"].mean(), 2)),
        "min_year": int(df["publication_year"].min()),
        "max_year": int(df["publication_year"].max()),
        "reading_levels": sorted(df["reading_level"].unique().tolist()),
    }
