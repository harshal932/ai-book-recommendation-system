"""Feedback management and persistence module.

Important Academic Notice:
--------------------------
Standard content-based filtering models do not automatically retrain in real time
upon feedback ingestion without explicit reinforcement loops.

In this system:
1. Feedback is persistently recorded to `data/feedback.csv` for audit and evaluation.
2. An explicit, transparent feedback modifier function (`compute_feedback_boost`)
   is provided for opt-in reranking, adjusting recommendation weights based on
   historical user satisfaction ratios.
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd


FEEDBACK_COLUMNS = [
    "feedback_id",
    "timestamp",
    "book_id",
    "title",
    "query_mode",
    "helpful",
    "comments",
]


def get_default_feedback_path() -> Path:
    """Returns path to data/feedback.csv."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "feedback.csv"


def ensure_feedback_file_exists(feedback_path: Optional[str | Path] = None) -> Path:
    """Ensures data/feedback.csv exists with proper schema."""
    path = Path(feedback_path) if feedback_path else get_default_feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        empty_df = pd.DataFrame(columns=FEEDBACK_COLUMNS)
        empty_df.to_csv(path, index=False, encoding="utf-8")
    return path


def record_feedback(
    book_id: str,
    title: str,
    helpful: bool,
    query_mode: str = "hybrid",
    comments: str = "",
    feedback_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Persist a feedback entry to data/feedback.csv.

    Args:
        book_id: Unique book ID.
        title: Book title.
        helpful: True if marked helpful, False if unhelpful.
        query_mode: Mode used ('hybrid', 'content_based', 'rule_based').
        comments: Optional free text comments from user.
        feedback_path: Optional custom path to feedback CSV.

    Returns:
        Dictionary of the recorded feedback entry.
    """
    path = ensure_feedback_file_exists(feedback_path)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_id = f"FB_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

    entry = {
        "feedback_id": feedback_id,
        "timestamp": timestamp_str,
        "book_id": str(book_id).strip(),
        "title": str(title).strip(),
        "query_mode": str(query_mode).strip(),
        "helpful": 1 if helpful else 0,
        "comments": str(comments).strip().replace("\n", " "),
    }

    new_row = pd.DataFrame([entry])

    # Append to CSV
    new_row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    return entry


def load_feedback(feedback_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load and return all recorded user feedback as a DataFrame."""
    path = ensure_feedback_file_exists(feedback_path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
        # Ensure correct columns exist
        for col in FEEDBACK_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df["helpful"] = pd.to_numeric(df["helpful"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def get_feedback_metrics(feedback_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Calculate aggregate feedback metrics for analytics display."""
    df = load_feedback(feedback_path)
    if df.empty:
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "helpful_ratio": 0.0,
            "recent_entries": [],
        }

    total = len(df)
    helpful_count = int((df["helpful"] == 1).sum())
    unhelpful_count = total - helpful_count
    ratio = round((helpful_count / total) * 100, 1) if total > 0 else 0.0

    recent = df.sort_values(by="timestamp", ascending=False).head(10).to_dict(orient="records")

    return {
        "total_feedback": total,
        "helpful_count": helpful_count,
        "unhelpful_count": unhelpful_count,
        "helpful_ratio": ratio,
        "recent_entries": recent,
    }


def compute_feedback_boost(book_id: str, feedback_path: Optional[str | Path] = None) -> float:
    """Calculate an empirical feedback score modifier for a book.

    Formula:
    If a book has received user feedback:
    boost = 0.05 * (positive_votes - negative_votes) / total_votes_for_book
    Clamped strictly to [-0.05, +0.05].

    Args:
        book_id: Target book ID.
        feedback_path: Optional path to feedback CSV.

    Returns:
        Float modifier between -0.05 and +0.05.
    """
    df = load_feedback(feedback_path)
    if df.empty:
        return 0.0

    book_fb = df[df["book_id"] == str(book_id).strip()]
    if book_fb.empty:
        return 0.0

    pos = (book_fb["helpful"] == 1).sum()
    neg = (book_fb["helpful"] == 0).sum()
    total = pos + neg

    if total == 0:
        return 0.0

    net_rate = (pos - neg) / total
    boost = 0.05 * net_rate
    return round(float(boost), 4)
