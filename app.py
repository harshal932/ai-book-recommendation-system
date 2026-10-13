"""AI-Based Intelligent Book Recommendation System
Using Content-Based Filtering and Rule-Based Reasoning.

Main Streamlit Application.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure src modules are resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (
    load_books_dataset,
    get_dataset_statistics,
    REQUIRED_COLUMNS,
)
from src.recommender import ContentBasedRecommender
from src.rule_engine import UserPreferences, RuleEngine, HybridRecommender
from src.feedback import (
    record_feedback,
    load_feedback,
    get_feedback_metrics,
    compute_feedback_boost,
)
from src.analytics import (
    plot_genre_distribution,
    plot_rating_distribution,
    plot_reading_level_distribution,
    plot_feedback_satisfaction,
)

# Page configuration
st.set_page_config(
    page_title="AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich scholastic theme and responsive cards
CUSTOM_CSS = """
<style>
    /* Global scholastic & editorial styling */
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 900px;
    }

    /* Metric Card */
    .stat-card {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.06);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Book Card */
    .book-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #2563eb;
    }
    .book-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .book-card-author {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 0.6rem;
    }
    .book-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-genre {
        background-color: #dbeafe;
        color: #1d4ed8;
    }
    .badge-level {
        background-color: #fef3c7;
        color: #b45309;
    }
    .badge-year {
        background-color: #f1f5f9;
        color: #475569;
    }
    .badge-rating {
        background-color: #dcfce7;
        color: #15803d;
    }
    .score-banner {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.9rem;
        color: #166534;
        font-weight: 600;
        margin: 0.6rem 0;
    }
    .disclaimer-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading book catalog...")
def get_cached_books_data() -> pd.DataFrame:
    """Load and cache the books dataset."""
    return load_books_dataset()


def get_cached_models(df: pd.DataFrame):
    """Initialize recommender models (saved in session state for fast reuse)."""
    if "content_rec" not in st.session_state or "hybrid_rec" not in st.session_state:
        st.session_state.content_rec = ContentBasedRecommender(df)
        st.session_state.hybrid_rec = HybridRecommender(df, st.session_state.content_rec)
    return st.session_state.content_rec, st.session_state.hybrid_rec


# ==============================================================================
# DATASET INITIALIZATION & ERROR HANDLING
# ==============================================================================
try:
    books_df = get_cached_books_data()
    content_recommender, hybrid_recommender = get_cached_models(books_df)
    stats = get_dataset_statistics(books_df)
except Exception as err:
    st.error(f"❌ Failed to load dataset: {err}")
    st.info("Ensure `data/books.csv` exists and matches the required schema.")
    st.stop()


# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM METRICS
# ==============================================================================
with st.sidebar:
    st.markdown("### 📚 Navigation")
    menu_choice = st.radio(
        "Select Module:",
        [
            "🏠 Dashboard & Overview",
            "⚡ Hybrid Recommender",
            "🔍 Book-to-Book Similarity",
            "📖 Catalog Explorer",
            "📊 Analytics & Insights",
            "🎓 Viva & Algorithm Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📌 System Status")
    st.markdown(f"**Total Books:** `{stats['total_books']}`")
    st.markdown(f"**Unique Genres:** `{stats['unique_genres']}`")
    st.markdown(f"**Catalog Mean Rating:** `★ {stats['avg_rating']}`")
    
    # Reload dataset action
    if st.button("🔄 Refresh Catalog Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("content_rec", None)
        st.session_state.pop("hybrid_rec", None)
        st.success("Cache refreshed!")
        st.rerun()

    st.markdown("---")
    st.caption("College Mini-Project in Artificial Intelligence\nContent-Based Filtering + Rule Reasoning")


# ==============================================================================
# 1. DASHBOARD & OVERVIEW
# ==============================================================================
if menu_choice == "🏠 Dashboard & Overview":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI-Based Intelligent Book Recommendation System</div>
            <div class="hero-subtitle">
                An explainable, hybrid recommender architecture combining <strong>Content-Based Filtering</strong> 
                (TF-IDF term vectorization and Cosine Similarity) with transparent <strong>Rule-Based Reasoning</strong> 
                (declarative IF-THEN production rules and weighted score fusion).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key statistics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{stats['total_books']}</div>
                <div class="stat-label">Verified Titles</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{stats['unique_genres']}</div>
                <div class="stat-label">Diverse Genres</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{stats['unique_authors']}</div>
                <div class="stat-label">Distinct Authors</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        fb_metrics = get_feedback_metrics()
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{fb_metrics['helpful_ratio']}%</div>
                <div class="stat-label">User Satisfaction</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Capabilities & How it works
    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        st.markdown("### 🎯 System Features & Highlights")
        st.markdown("""
        - **Content-Based Vector Space Model:** Converts titles, descriptions, genres, and metadata into TF-IDF vectors (unigrams + bigrams), evaluating thematic affinity with high-dimensional Cosine Similarity.
        - **Declarative Rule Engine:** Evaluates explicit IF-THEN rules for reading levels, user genre interests, publication era preferences, and quality thresholds.
        - **Configurable Score Fusion:** Interactively blends content similarity and rule weights with strictly normalized ranking:
          $$\\text{Hybrid Score} = w_{\\text{content}} \\times \\text{Sim}_{\\text{norm}} + w_{\\text{rule}} \\times \\text{Rule}_{\\text{norm}}$$
        - **Complete Explainability Audit:** Displays exact lexical overlap terms and active rule activations justifying each recommendation.
        - **Persistent User Feedback:** Captures real-time user ratings with empirical reranking support.
        """)

    with col_right:
        st.markdown("### 🏷️ Catalog Diversity by Genre")
        fig_genre = plot_genre_distribution(books_df)
        if fig_genre:
            st.pyplot(fig_genre)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.info(
        "💡 **To get recommendations immediately:** Open **'⚡ Hybrid Recommender'** in the sidebar. "
        "Select your preferred genres, target reading level, and adjust the Content vs. Rule weighting slider!"
    )


# ==============================================================================
# 2. HYBRID RECOMMENDER (CONTENT-BASED + RULE-BASED)
# ==============================================================================
elif menu_choice == "⚡ Hybrid Recommender":
    st.markdown("## ⚡ Intelligent Hybrid Recommendation Engine")
    st.markdown(
        "Configure your reading interests and rule constraints. "
        "The engine calculates both TF-IDF content similarity and declarative rule scores, combining them with full transparency."
    )

    with st.expander("🛠️ Recommender Configuration & User Preference Inputs", expanded=True):
        tab_input_type = st.radio(
            "Recommendation Seed Type:",
            ["By User Criteria & Preferences Only", "By Seed Book + Preferences", "By Free-Text Search Prompt + Preferences"],
            horizontal=True,
        )

        seed_book_id: Optional[str] = None
        user_query_text: Optional[str] = None

        if tab_input_type == "By Seed Book + Preferences":
            book_titles_map = {f"{row['title']} ({row['author']})": row["book_id"] for _, row in books_df.iterrows()}
            selected_book_title = st.selectbox(
                "Select a Seed Book you enjoyed:",
                options=list(book_titles_map.keys()),
                index=0,
            )
            seed_book_id = book_titles_map[selected_book_title]
            seed_row = books_df[books_df["book_id"] == seed_book_id].iloc[0]
            st.caption(f"**Seed Book Genre:** {seed_row['genre']} | **Reading Level:** {seed_row['reading_level']} | **Rating:** ★ {seed_row['rating']}")

        elif tab_input_type == "By Free-Text Search Prompt + Preferences":
            user_query_text = st.text_input(
                "Describe topics or themes you are looking for:",
                value="algorithms data structures and artificial intelligence",
                placeholder="e.g. quantum mechanics cosmology astrophysics",
            )

        st.markdown("#### 🎯 Rule Engine Criteria (IF-THEN Rules)")
        c1, c2, c3 = st.columns(3)

        with c1:
            selected_genres = st.multiselect(
                "Target Genres (Rule 1: Genre Match):",
                options=stats["genres_list"],
                default=[],
                help="Select one or more genres to prioritize.",
            )

        with c2:
            selected_reading_level = st.selectbox(
                "Target Reading Level (Rule 2: Level Match):",
                options=["Any", "Beginner", "Intermediate", "Advanced"],
                index=0,
            )

        with c3:
            era_preference = st.selectbox(
                "Publication Era (Rule 4: Era Match):",
                options=["All Eras", "Classic (< 1980)", "Modern (>= 1980)", "Contemporary (>= 2010)"],
                index=0,
            )
            era_code = "All"
            if "Classic" in era_preference:
                era_code = "Classic"
            elif "Modern" in era_preference:
                era_code = "Modern"
            elif "Contemporary" in era_preference:
                era_code = "Contemporary"

        c4, c5 = st.columns([1, 1])
        with c4:
            min_rating_threshold = st.slider(
                "Minimum Rating Threshold (Rule 3: Quality Filter):",
                min_value=0.0,
                max_value=5.0,
                value=4.0,
                step=0.1,
            )
        with c5:
            topic_keywords_raw = st.text_input(
                "Specific Topic Keywords (Rule 5):",
                placeholder="e.g. stoic, python, biology, startup (comma separated)",
            )
            topic_keywords = [k.strip() for k in topic_keywords_raw.split(",") if k.strip()]

        st.markdown("#### ⚖️ Scoring Weight Fusion")
        w_col1, w_col2, w_col3 = st.columns([2, 1, 1])
        with w_col1:
            content_weight_pct = st.slider(
                "Content Similarity Weight vs. Rule Engine Weight:",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="100% = Pure TF-IDF Content Similarity. 0% = Pure Rule-Based Engine.",
            )
            w_content = content_weight_pct / 100.0
            w_rule = (100 - content_weight_pct) / 100.0
            st.caption(f"Current Fusion Ratio: **{int(w_content*100)}% Content-Based** + **{int(w_rule*100)}% Rule-Based**")

        with w_col2:
            top_n_choice = st.selectbox("Number of Recommendations:", [3, 5, 10], index=1)

        with w_col3:
            use_feedback_boost = st.checkbox(
                "Include Historical Feedback Adjustment",
                value=False,
                help="Applies transparent +/- 0.05 modifier based on past user satisfaction data.",
            )

    # Run recommendation
    reading_lvl_arg = None if selected_reading_level == "Any" else selected_reading_level
    prefs = UserPreferences(
        preferred_genres=selected_genres,
        reading_level=reading_lvl_arg,
        min_rating=min_rating_threshold,
        era_preference=era_code,
        topic_keywords=topic_keywords,
    )

    recommendations = hybrid_recommender.recommend_hybrid(
        prefs=prefs,
        seed_book_id=seed_book_id,
        query_text=user_query_text,
        w_content=w_content,
        w_rule=w_rule,
        top_n=top_n_choice,
        exclude_seed=True,
    )

    # Optional feedback adjustment
    if use_feedback_boost and recommendations:
        for item in recommendations:
            boost = compute_feedback_boost(item["book_id"])
            item["hybrid_score"] = round(min(1.0, max(0.0, item["hybrid_score"] + boost)), 4)
            if boost != 0.0:
                item["rules_fired"].append(f"Feedback Modifier: Adjusted by {boost:+.3f} from past user reviews.")
        # Re-sort after boost
        recommendations.sort(key=lambda x: (x["hybrid_score"], x["rating"]), reverse=True)

    st.markdown("### 📋 Recommendation Results")

    if not recommendations:
        st.warning("No books matched the current criteria. Try loosening the rating threshold or genre constraints.")
    else:
        for idx, rec in enumerate(recommendations, start=1):
            with st.container():
                st.markdown(
                    f"""
                    <div class="book-card">
                        <div class="book-card-title">#{idx} {rec['title']}</div>
                        <div class="book-card-author">By <strong>{rec['author']}</strong></div>
                        <div>
                            <span class="book-badge badge-genre">📖 {rec['genre']}</span>
                            <span class="book-badge badge-level">🎯 {rec['reading_level']}</span>
                            <span class="book-badge badge-rating">★ {rec['rating']}</span>
                            <span class="book-badge badge-year">📅 {rec['publication_year']}</span>
                        </div>
                        <div class="score-banner">
                            ✨ Hybrid Match Score: {round(rec['hybrid_score'] * 100, 1)}% 
                            &nbsp;|&nbsp; Content Sim: {round(rec['content_similarity_score'] * 100, 1)}%
                            &nbsp;|&nbsp; Rule Match: {round(rec['rule_score'] * 100, 1)}%
                        </div>
                        <p style="color: #334155; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.5;">
                            {rec['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Explainability & Audit Trail
                with st.expander(f"🔍 Why was '{rec['title']}' recommended? (Explainability & Rules Audit)"):
                    st.markdown("**Algorithmic Breakdown:**")
                    st.write(f"- {rec['explanation']}")
                    st.markdown("**Active IF-THEN Rules Fired:**")
                    for rule in rec["rules_fired"]:
                        st.markdown(f"- `{rule}`")

                # Feedback widget
                fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 4])
                with fb_c1:
                    if st.button("👍 Helpful", key=f"helpful_{rec['book_id']}_{idx}"):
                        record_feedback(
                            book_id=rec["book_id"],
                            title=rec["title"],
                            helpful=True,
                            query_mode="hybrid",
                            comments="Marked helpful by user",
                        )
                        st.toast(f"Saved feedback for '{rec['title']}'! Thank you.", icon="✅")
                with fb_c2:
                    if st.button("👎 Not Helpful", key=f"unhelpful_{rec['book_id']}_{idx}"):
                        record_feedback(
                            book_id=rec["book_id"],
                            title=rec["title"],
                            helpful=False,
                            query_mode="hybrid",
                            comments="Marked unhelpful by user",
                        )
                        st.toast(f"Feedback recorded for '{rec['title']}'.", icon="ℹ️")

                st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px dashed #e2e8f0;'/>", unsafe_allow_html=True)

        st.markdown(
            """<div class="disclaimer-box">
                ℹ️ <strong>Academic Transparency Note:</strong>
                Scores displayed are mathematical similarity and rule compliance metrics within the vector space [0.0 - 1.0].
                They represent algorithmic affinity, not guaranteed human subjective enjoyment.
            </div>""",
            unsafe_allow_html=True,
        )


# ==============================================================================
# 3. BOOK-TO-BOOK SIMILARITY FINDER (CONTENT-BASED ONLY)
# ==============================================================================
elif menu_choice == "🔍 Book-to-Book Similarity":
    st.markdown("## 🔍 Book-to-Book Content Similarity Explorer")
    st.markdown(
        "Demonstrates pure **Content-Based Filtering** using TF-IDF term vectorization and Cosine Similarity. "
        "Select any book in the catalog to discover titles with the closest textual and thematic profile."
    )

    book_map = {f"{row['title']} ({row['author']}) - [{row['genre']}]": row["book_id"] for _, row in books_df.iterrows()}
    chosen_label = st.selectbox("Choose a Target Book:", options=list(book_map.keys()), index=0)
    target_id = book_map[chosen_label]
    target_book = books_df[books_df["book_id"] == target_id].iloc[0]

    # Target Book Summary Card
    st.markdown(
        f"""
        <div style="background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 10px; padding: 1.2rem; margin: 1rem 0;">
            <h4 style="margin: 0; color: #0369a1;">Selected Reference Book: {target_book['title']}</h4>
            <p style="margin: 0.25rem 0; color: #0c4a6e;"><strong>Author:</strong> {target_book['author']} | <strong>Genre:</strong> {target_book['genre']} | <strong>Rating:</strong> ★ {target_book['rating']}</p>
            <p style="margin: 0; font-size: 0.9rem; color: #075985;">{target_book['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sim1, col_sim2 = st.columns([1, 3])
    with col_sim1:
        sim_top_n = st.selectbox("Number of Similar Books:", [3, 5, 10], index=1)
    
    similar_books = content_recommender.get_similar_books(
        book_id=target_id,
        top_n=sim_top_n,
        exclude_self=True,
    )

    st.markdown("### 📚 Most Similar Titles in Catalog")
    if not similar_books:
        st.info("No similar titles found.")
    else:
        for idx, item in enumerate(similar_books, start=1):
            st.markdown(
                f"""
                <div class="book-card" style="border-left-color: #0284c7;">
                    <div class="book-card-title">#{idx} {item['title']}</div>
                    <div class="book-card-author">By <strong>{item['author']}</strong></div>
                    <div>
                        <span class="book-badge badge-genre">{item['genre']}</span>
                        <span class="book-badge badge-level">{item['reading_level']}</span>
                        <span class="book-badge badge-rating">★ {item['rating']}</span>
                        <span class="book-badge badge-year">{item['publication_year']}</span>
                    </div>
                    <div class="score-banner" style="background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8;">
                        Cosine Similarity: {round(item['similarity_score'] * 100, 1)}% 
                        &nbsp;|&nbsp; Common Overlapping Tokens: <code>{', '.join(item['common_terms']) if item['common_terms'] else 'General theme'}</code>
                    </div>
                    <p style="color: #334155; font-size: 0.92rem; margin-top: 0.5rem;">
                        {item['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 4. CATALOG EXPLORER
# ==============================================================================
elif menu_choice == "📖 Catalog Explorer":
    st.markdown("## 📖 Book Catalog & Discovery")
    st.markdown("Search, filter, and inspect verified titles in the dataset.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        search_query = st.text_input("🔍 Search by Title, Author, or Keyword:", placeholder="e.g. Orwell, habits, machine learning...")
    with c2:
        genre_filter = st.selectbox("Genre Filter:", ["All Genres"] + stats["genres_list"])
    with c3:
        level_filter = st.selectbox("Reading Level:", ["All Levels", "Beginner", "Intermediate", "Advanced"])
    with c4:
        sort_by = st.selectbox("Sort By:", ["Rating (High to Low)", "Year (Newest First)", "Title (A-Z)"])

    filtered_df = books_df.copy()

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["title"].str.lower().str.contains(q)
            | filtered_df["author"].str.lower().str.contains(q)
            | filtered_df["description"].str.lower().str.contains(q)
        ]

    if genre_filter != "All Genres":
        filtered_df = filtered_df[filtered_df["genre"] == genre_filter]

    if level_filter != "All Levels":
        filtered_df = filtered_df[filtered_df["reading_level"] == level_filter]

    if sort_by == "Rating (High to Low)":
        filtered_df = filtered_df.sort_values(by="rating", ascending=False)
    elif sort_by == "Year (Newest First)":
        filtered_df = filtered_df.sort_values(by="publication_year", ascending=False)
    elif sort_by == "Title (A-Z)":
        filtered_df = filtered_df.sort_values(by="title", ascending=True)

    st.markdown(f"**Found `{len(filtered_df)}` books matching your filters:**")

    # Render table or cards
    for _, row in filtered_df.head(20).iterrows():
        with st.container():
            st.markdown(
                f"""
                <div class="book-card" style="border-left-color: #64748b;">
                    <div class="book-card-title">{row['title']}</div>
                    <div class="book-card-author">By <strong>{row['author']}</strong></div>
                    <div>
                        <span class="book-badge badge-genre">{row['genre']}</span>
                        <span class="book-badge badge-level">{row['reading_level']}</span>
                        <span class="book-badge badge-rating">★ {row['rating']}</span>
                        <span class="book-badge badge-year">📅 {row['publication_year']}</span>
                        <span class="book-badge badge-year">ID: {row['book_id']}</span>
                    </div>
                    <p style="color: #475569; font-size: 0.9rem; margin-top: 0.5rem;">
                        {row['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if len(filtered_df) > 20:
        st.caption(f"Showing first 20 of {len(filtered_df)} matching books.")


# ==============================================================================
# 5. ANALYTICS & INSIGHTS
# ==============================================================================
elif menu_choice == "📊 Analytics & Insights":
    st.markdown("## 📊 Catalog & User Feedback Analytics")
    st.markdown("Visual explorations of the dataset distributions and recorded user feedback metrics.")

    tab1, tab2 = st.tabs(["📚 Catalog Distributions", "💬 Feedback Analytics"])

    with tab1:
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown("#### Genre Frequency")
            fig1 = plot_genre_distribution(books_df)
            if fig1:
                st.pyplot(fig1)

        with a_col2:
            st.markdown("#### Reading Level Distribution")
            fig2 = plot_reading_level_distribution(books_df)
            if fig2:
                st.pyplot(fig2)

        st.markdown("#### Catalog Rating Distribution")
        fig3 = plot_rating_distribution(books_df)
        if fig3:
            st.pyplot(fig3)

    with tab2:
        fb_df = load_feedback()
        fb_metrics = get_feedback_metrics()

        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            st.metric("Total User Votes", fb_metrics["total_feedback"])
        with f_c2:
            st.metric("Helpful (👍)", fb_metrics["helpful_count"])
        with f_c3:
            st.metric("Satisfaction Rate", f"{fb_metrics['helpful_ratio']}%")

        st.markdown("<br>", unsafe_allow_html=True)
        fb_fig = plot_feedback_satisfaction(fb_df)
        if fb_fig:
            st.pyplot(fb_fig)

        st.markdown("#### Recent Feedback Log")
        if not fb_df.empty:
            st.dataframe(fb_df.tail(15), use_container_width=True)
        else:
            st.info("No feedback has been recorded yet.")


# ==============================================================================
# 6. VIVA & ALGORITHM GUIDE
# ==============================================================================
elif menu_choice == "🎓 Viva & Algorithm Guide":
    st.markdown("## 🎓 Academic Viva & Technical Explanation Guide")
    st.markdown(
        "Structured summary for undergraduate Artificial Intelligence viva examinations, demonstrations, and practical evaluations."
    )

    with st.expander("1. System Architecture & Methodology", expanded=True):
        st.markdown("""
        The system implements a **Hybrid AI Recommender System** with two complementary reasoning paradigms:
        1. **Content-Based Filtering (Inductive / Vector Space Model):**
           - Represents metadata as token frequency distributions using **TF-IDF**.
           - Calculates similarity as the cosine of the angle between high-dimensional vectors:
             $$\\text{Cosine Similarity}(\\vec{u}, \\vec{v}) = \\frac{\\vec{u} \\cdot \\vec{v}}{\\|\\vec{u}\\| \\|\\vec{v}\\|}$$
        2. **Rule-Based Reasoning (Deductive / Production System):**
           - Implements explicit declarative IF-THEN production rules representing user domain constraints.
           - Evaluates genre preference, target reading level, publication era, and rating constraints.
           - Generates a full audit trail for explainability.
        3. **Hybrid Score Fusion:**
           - Normalizes both components to $[0, 1]$ and computes a user-weighted convex combination:
             $$S_{\\text{hybrid}} = w_c \\cdot S_{\\text{content}} + w_r \\cdot S_{\\text{rule}}$$
        """)

    with st.expander("2. Quick Viva Cheatsheet (Top 5 Questions & Direct Answers)", expanded=True):
        st.markdown("""
        **Q1: What is the primary difference between Content-Based and Collaborative Filtering?**  
        *Answer:* Content-Based Filtering recommends items based on the attributes and metadata of items the user has liked or requested (e.g., text descriptions, genre, author). Collaborative Filtering relies on collective user behavior patterns and rating overlaps without necessarily understanding item contents.

        **Q2: What is the Cold-Start Problem and how does this project handle it?**  
        *Answer:* The Cold-Start problem occurs when a new user or new item enters the system with zero historical rating data. Collaborative filtering fails here. Our system solves it because Content-Based and Rule-Based methods require *no* prior rating history—they function immediately based on item metadata and user preference constraints.

        **Q3: Why use Cosine Similarity instead of Euclidean Distance?**  
        *Answer:* Cosine similarity measures the orientation (angle) rather than magnitude. In text analysis, longer descriptions have higher term frequencies and larger vector magnitudes. Euclidean distance would penalize length differences, whereas Cosine Similarity correctly measures thematic alignment regardless of description length.

        **Q4: How does TF-IDF prevent common words from dominating?**  
        *Answer:* Term Frequency (TF) counts word occurrences in a document, while Inverse Document Frequency (IDF) penalizes words that appear across many documents (like 'book' or 'story'). IDF is $\\log(\\frac{N}{df})$, so universally frequent words receive near-zero weight.

        **Q5: Is this system a black box?**  
        *Answer:* No. It is fully explainable. Every recommendation displays an explainability audit trail showing which exact IF-THEN rules fired, rule score point allocations, and highest-weighted overlapping TF-IDF terms.
        """)

    st.info("📖 For comprehensive preparation, consult `docs/project_report.md`, `docs/algorithm.md`, and `docs/viva_questions.md` included in the project repository.")
