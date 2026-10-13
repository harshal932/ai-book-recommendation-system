"""Rule-Based Reasoning Engine and Hybrid Score Fusion.

Important Academic Notice:
--------------------------
This engine executes explicit, declarative IF-THEN production rules over structured
attributes (genre, reading level, minimum rating, publication era, and topic keywords).
It audits and logs every rule activation to guarantee full explainability.

Combined Ranking Formula:
-------------------------
Hybrid Score = (w_content * Content_Similarity_Normalized) + (w_rule * Rule_Score_Normalized)
Both components are strictly normalized to the [0.0, 1.0] interval.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserPreferences:
    """User preference container for rule evaluation."""

    def __init__(
        self,
        preferred_genres: Optional[List[str]] = None,
        reading_level: Optional[str] = None,
        min_rating: float = 0.0,
        era_preference: Optional[str] = None,  # 'All', 'Classic' (<1980), 'Modern' (>=1980), 'Contemporary' (>=2010)
        topic_keywords: Optional[List[str]] = None,
    ):
        self.preferred_genres: Set[str] = {g.strip().title() for g in (preferred_genres or []) if g.strip()}
        self.reading_level: Optional[str] = reading_level.strip().capitalize() if reading_level else None
        self.min_rating: float = max(0.0, min(5.0, float(min_rating)))
        self.era_preference: str = era_preference.strip().title() if era_preference else "All"
        self.topic_keywords: List[str] = [k.strip().lower() for k in (topic_keywords or []) if k.strip()]

    def is_empty(self) -> bool:
        """Returns True if user has selected no specific criteria."""
        return (
            len(self.preferred_genres) == 0
            and self.reading_level is None
            and self.min_rating <= 0.0
            and self.era_preference in ("All", "")
            and len(self.topic_keywords) == 0
        )


class RuleEngine:
    """Evaluates IF-THEN rules and produces explainable rule scores."""

    # Maximum theoretical weights per rule category for normalization
    WEIGHT_GENRE = 0.35
    WEIGHT_READING_LEVEL = 0.25
    WEIGHT_RATING = 0.20
    WEIGHT_ERA = 0.10
    WEIGHT_TOPIC = 0.10

    TOTAL_POSSIBLE_WEIGHT = WEIGHT_GENRE + WEIGHT_READING_LEVEL + WEIGHT_RATING + WEIGHT_ERA + WEIGHT_TOPIC

    def evaluate_book(self, book: Dict[str, Any], prefs: UserPreferences) -> Tuple[float, List[str]]:
        """Evaluate a single book against user preferences using IF-THEN rules.

        Args:
            book: Dictionary of book fields (genre, rating, reading_level, publication_year, etc.).
            prefs: UserPreferences object.

        Returns:
            Tuple of:
                - normalized_rule_score: Float between 0.0 and 1.0.
                - audit_trail: List of human-readable explanations of fired rules.
        """
        raw_score = 0.0
        audit_trail: List[str] = []

        if prefs.is_empty():
            # Baseline neutral score if no preferences are provided
            return 0.50, ["Default Rule: Neutral baseline score applied (no user filters selected)."]

        # RULE 1: Genre Preference Rule
        # IF the book's genre is in the user's selected preferred genres, THEN award genre bonus
        book_genre = str(book.get("genre", "")).strip().title()
        if prefs.preferred_genres:
            if book_genre in prefs.preferred_genres:
                raw_score += self.WEIGHT_GENRE
                audit_trail.append(f"Rule [Genre Match]: Matches your preferred genre '{book_genre}' (+{self.WEIGHT_GENRE:.2f}).")
            else:
                audit_trail.append(f"Rule [Genre Mismatch]: Book genre '{book_genre}' was not among selected genres.")
        else:
            # If user didn't specify genre, assign neutral fractional credit
            raw_score += self.WEIGHT_GENRE * 0.5
            audit_trail.append("Rule [Genre Neutral]: No genre filter specified (50% neutral credit).")

        # RULE 2: Reading Level Match Rule
        # IF reading level matches exactly, award full points; adjacent level gets partial
        book_level = str(book.get("reading_level", "")).strip().capitalize()
        if prefs.reading_level:
            if book_level == prefs.reading_level:
                raw_score += self.WEIGHT_READING_LEVEL
                audit_trail.append(
                    f"Rule [Reading Level Match]: Exact match for '{book_level}' reading level (+{self.WEIGHT_READING_LEVEL:.2f})."
                )
            elif (
                (prefs.reading_level == "Beginner" and book_level == "Intermediate")
                or (prefs.reading_level == "Intermediate" and book_level in ("Beginner", "Advanced"))
                or (prefs.reading_level == "Advanced" and book_level == "Intermediate")
            ):
                partial = self.WEIGHT_READING_LEVEL * 0.4
                raw_score += partial
                audit_trail.append(
                    f"Rule [Reading Level Near]: Adjacent reading level '{book_level}' vs requested '{prefs.reading_level}' (+{partial:.2f})."
                )
            else:
                audit_trail.append(
                    f"Rule [Reading Level Mismatch]: Book level '{book_level}' deviates from requested '{prefs.reading_level}'."
                )
        else:
            raw_score += self.WEIGHT_READING_LEVEL * 0.5
            audit_trail.append("Rule [Reading Level Neutral]: No reading level specified (neutral credit).")

        # RULE 3: Rating Quality Threshold Rule
        # IF book rating >= requested min_rating, award score based on normalized quality
        book_rating = float(book.get("rating", 4.0))
        if prefs.min_rating > 0.0:
            if book_rating >= prefs.min_rating:
                # Scaled reward based on how high the rating is
                scale = (book_rating / 5.0) * self.WEIGHT_RATING
                raw_score += scale
                audit_trail.append(
                    f"Rule [Rating Criterion Met]: Book rating {book_rating} exceeds threshold {prefs.min_rating} (+{scale:.2f})."
                )
            else:
                audit_trail.append(
                    f"Rule [Rating Penalty]: Book rating {book_rating} is below requested threshold {prefs.min_rating}."
                )
        else:
            scale = (book_rating / 5.0) * self.WEIGHT_RATING
            raw_score += scale
            audit_trail.append(f"Rule [General Rating Quality]: Rating of {book_rating}/5.0 awarded (+{scale:.2f}).")

        # RULE 4: Publication Era Rule
        # IF publication year falls into requested era
        pub_year = int(book.get("publication_year", 2000))
        if prefs.era_preference == "Classic":
            if pub_year < 1980:
                raw_score += self.WEIGHT_ERA
                audit_trail.append(f"Rule [Era Match]: Published in {pub_year} (<1980 Classic) (+{self.WEIGHT_ERA:.2f}).")
            else:
                audit_trail.append(f"Rule [Era Non-Match]: Published in {pub_year} (not classic).")
        elif prefs.era_preference == "Modern":
            if pub_year >= 1980:
                raw_score += self.WEIGHT_ERA
                audit_trail.append(f"Rule [Era Match]: Published in {pub_year} (>=1980 Modern) (+{self.WEIGHT_ERA:.2f}).")
            else:
                audit_trail.append(f"Rule [Era Non-Match]: Published in {pub_year} (pre-1980).")
        elif prefs.era_preference == "Contemporary":
            if pub_year >= 2010:
                raw_score += self.WEIGHT_ERA
                audit_trail.append(f"Rule [Era Match]: Published in {pub_year} (>=2010 Contemporary) (+{self.WEIGHT_ERA:.2f}).")
            else:
                audit_trail.append(f"Rule [Era Non-Match]: Published in {pub_year} (pre-2010).")
        else:
            raw_score += self.WEIGHT_ERA * 0.5
            audit_trail.append("Rule [Era Neutral]: Any publication era accepted.")

        # RULE 5: Topic Keywords Match Rule
        # IF description contains user topic keywords
        desc_lower = str(book.get("description", "")).lower()
        title_lower = str(book.get("title", "")).lower()
        if prefs.topic_keywords:
            matched_keywords = [
                kw for kw in prefs.topic_keywords if kw in desc_lower or kw in title_lower
            ]
            if matched_keywords:
                fraction = min(1.0, len(matched_keywords) / len(prefs.topic_keywords))
                kw_score = fraction * self.WEIGHT_TOPIC
                raw_score += kw_score
                audit_trail.append(
                    f"Rule [Topic Match]: Found keywords {matched_keywords} in book metadata (+{kw_score:.2f})."
                )
            else:
                audit_trail.append("Rule [Topic Non-Match]: No specified topic keywords found.")
        else:
            raw_score += self.WEIGHT_TOPIC * 0.5
            audit_trail.append("Rule [Topic Neutral]: No specific topic keywords specified.")

        # Normalize to [0.0, 1.0]
        normalized_score = max(0.0, min(1.0, raw_score / self.TOTAL_POSSIBLE_WEIGHT))
        return round(normalized_score, 4), audit_trail


class HybridRecommender:
    """Fuses Content-Based Similarity with Rule-Based Reasoning."""

    def __init__(self, books_df: pd.DataFrame, content_recommender):
        """Initialize hybrid engine with dataset and fitted content engine.

        Args:
            books_df: Validated DataFrame of books.
            content_recommender: Instance of ContentBasedRecommender.
        """
        self.df = books_df.copy().reset_index(drop=True)
        self.content_engine = content_recommender
        self.rule_engine = RuleEngine()

    def recommend_hybrid(
        self,
        prefs: UserPreferences,
        seed_book_id: Optional[str] = None,
        query_text: Optional[str] = None,
        w_content: float = 0.50,
        w_rule: float = 0.50,
        top_n: int = 5,
        exclude_seed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compute hybrid ranked recommendations combining content similarity and rules.

        Args:
            prefs: UserPreferences containing selected criteria.
            seed_book_id: Optional book ID to compute content similarity against.
            query_text: Optional free-text search prompt.
            w_content: Weight assigned to content-based similarity [0.0 - 1.0].
            w_rule: Weight assigned to rule-based reasoning [0.0 - 1.0].
            top_n: Number of recommendations to return.
            exclude_seed: Whether to exclude the seed book from results.

        Returns:
            List of ranked recommendation dicts with hybrid score, score components, and explanations.
        """
        # Ensure weight sum is normalized to 1.0
        total_w = w_content + w_rule
        if total_w <= 0:
            w_c, w_r = 0.5, 0.5
        else:
            w_c = w_content / total_w
            w_r = w_rule / total_w

        # Step 1: Compute Content Similarity for all candidate books
        content_sims: Dict[str, float] = {}

        if seed_book_id and seed_book_id in self.content_engine.id_to_idx:
            # Book-to-book similarity vector
            target_idx = self.content_engine.id_to_idx[seed_book_id]
            vector = self.content_engine.cosine_sim_matrix[target_idx]
            for idx, row in self.df.iterrows():
                content_sims[row["book_id"]] = float(vector[idx])
        elif query_text and query_text.strip():
            # Query-to-book vector similarity
            query_vec = self.content_engine.vectorizer.transform([query_text.strip()])
            vector = cosine_similarity(query_vec, self.content_engine.tfidf_matrix).flatten()
            for idx, row in self.df.iterrows():
                content_sims[row["book_id"]] = float(vector[idx])
        else:
            # No content seed provided: content similarity is neutral across catalog
            for row in self.df["book_id"]:
                content_sims[row] = 0.0

        # Step 2: Evaluate Rules and Combine Scores
        results: List[Dict[str, Any]] = []

        for _, row in self.df.iterrows():
            b_id = row["book_id"]
            if exclude_seed and seed_book_id and b_id == seed_book_id:
                continue

            book_dict = row.to_dict()
            rule_score, audit_trail = self.rule_engine.evaluate_book(book_dict, prefs)
            sim_score = content_sims.get(b_id, 0.0)

            # Combined hybrid formula
            # Note: sim_score is in [0, 1], rule_score is in [0, 1]
            hybrid_score = (w_c * sim_score) + (w_r * rule_score)

            results.append({
                "book_id": b_id,
                "title": row["title"],
                "author": row["author"],
                "genre": row["genre"],
                "description": row["description"],
                "publication_year": int(row["publication_year"]),
                "rating": float(row["rating"]),
                "reading_level": row["reading_level"],
                "hybrid_score": round(float(hybrid_score), 4),
                "content_similarity_score": round(float(sim_score), 4),
                "rule_score": round(float(rule_score), 4),
                "weights_used": {"content": round(w_c, 2), "rule": round(w_r, 2)},
                "rules_fired": audit_trail,
                "explanation": (
                    f"Hybrid Score: {round(hybrid_score * 100, 1)}% "
                    f"({round(w_c * 100)}% Content Sim [{round(sim_score * 100, 1)}%] + "
                    f"{round(w_r * 100)}% Rule Match [{round(rule_score * 100, 1)}%])."
                ),
            })

        # Step 3: Sort descending by hybrid_score, tie-breaking by book rating
        results.sort(key=lambda x: (x["hybrid_score"], x["rating"]), reverse=True)

        return results[:top_n]
