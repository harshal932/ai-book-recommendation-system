"""Analytics and Visualization helpers for the Book Recommendation System.

Provides Matplotlib figure generators and summary tables for:
- Genre frequency distribution.
- Book rating distributions.
- Reading level counts.
- Publication era breakdown.
- User feedback satisfaction ratios.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt


# Use a clean, publication-ready plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_genre_distribution(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate a horizontal bar chart of books by genre."""
    if df.empty or "genre" not in df.columns:
        return None

    counts = df["genre"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(counts.index, counts.values, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)

    ax.set_title("Catalog Distribution by Genre", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Books", fontsize=10)
    ax.set_ylabel("Genre", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add count labels at the end of each bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate a histogram/KDE-style plot of book ratings."""
    if df.empty or "rating" not in df.columns:
        return None

    ratings = df["rating"].dropna()
    if len(ratings) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(
        ratings,
        bins=10,
        color="#10b981",
        edgecolor="#047857",
        alpha=0.8,
        rwidth=0.85,
    )

    ax.set_title("Distribution of Book Ratings", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rating (1.0 - 5.0 scale)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate mean rating line
    mean_val = float(ratings.mean())
    ax.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.legend(frameon=True)

    plt.tight_layout()
    return fig


def plot_reading_level_distribution(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate a donut chart or bar chart of reading levels."""
    if df.empty or "reading_level" not in df.columns:
        return None

    counts = df["reading_level"].value_counts()
    if counts.empty:
        return None

    colors = ["#6366f1", "#f59e0b", "#ec4899"]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[: len(counts)],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )

    for at in autotexts:
        at.set_color("black")
        at.set_fontweight("bold")

    ax.set_title("Books by Reading Level", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_feedback_satisfaction(feedback_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate a bar comparison of Helpful vs Not Helpful feedback."""
    if feedback_df.empty or "helpful" not in feedback_df.columns:
        return None

    helpful_counts = feedback_df["helpful"].value_counts()
    pos = int(helpful_counts.get(1, 0))
    neg = int(helpful_counts.get(0, 0))

    if pos + neg == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ["Helpful (Thumbs Up)", "Not Helpful (Thumbs Down)"]
    values = [pos, neg]
    colors = ["#10b981", "#ef4444"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#374151")
    ax.set_title("User Recommendation Satisfaction", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Total Votes", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{int(h)}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig
