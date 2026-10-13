"""Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.

Important Concept Explanation:
-------------------------------
Content-based filtering converts textual book metadata (description, genre, author,
and reading level) into high-dimensional numerical feature vectors using Term
Frequency-Inverse Document Frequency (TF-IDF). It then calculates the angular cosine
distance between vectors.

This measures lexical and thematic similarity in vector space. It does NOT possess
human-level reading comprehension or semantic consciousness; it ranks items purely
by mathematical token co-occurrence and importance weights.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """TF-IDF and Cosine Similarity recommendation engine for catalog books."""

    def __init__(self, df: pd.DataFrame):
        """Initialize and fit the TF-IDF vectorizer on the provided books DataFrame.

        Args:
            df: Cleaned books DataFrame containing title, genre, author, description, etc.
        """
        if df.empty:
            raise ValueError("Cannot initialize recommender with an empty DataFrame.")

        self.df = df.copy().reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2500,
            ngram_range=(1, 2),  # unigrams and bigrams
        )
        self._prepare_corpus_and_fit()

    def _prepare_corpus_and_fit(self) -> None:
        """Create a weighted metadata soup for each book and fit the TF-IDF matrix."""
        # Create a rich text soup combining genre, author, description, and reading level
        # Repeating genre and keywords slightly increases their thematic weight
        self.df["content_soup"] = (
            self.df["genre"] + " " +
            self.df["genre"] + " " +
            self.df["author"] + " " +
            self.df["reading_level"] + " level. " +
            self.df["description"]
        ).fillna("")

        # Fit and transform the corpus into TF-IDF sparse matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content_soup"])
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Mapping for fast lookup
        self.id_to_idx = {row["book_id"]: idx for idx, row in self.df.iterrows()}
        self.title_to_idx = {row["title"].lower().strip(): idx for idx, row in self.df.iterrows()}

    def get_similar_books(
        self,
        book_id: Optional[str] = None,
        title: Optional[str] = None,
        top_n: int = 5,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """Recommend books most similar to a chosen target book using cosine similarity.

        Args:
            book_id: Unique book ID (e.g. 'B001').
            title: Case-insensitive book title.
            top_n: Number of recommendations to return (default: 5).
            exclude_self: Whether to exclude the target book from results (default: True).

        Returns:
            List of recommendation dicts including book metadata, similarity score, and explanation.
        """
        target_idx: Optional[int] = None

        if book_id and book_id in self.id_to_idx:
            target_idx = self.id_to_idx[book_id]
        elif title and title.lower().strip() in self.title_to_idx:
            target_idx = self.title_to_idx[title.lower().strip()]

        if target_idx is None:
            return []

        # Get pairwise similarity scores for this book
        sim_scores = list(enumerate(self.cosine_sim_matrix[target_idx]))

        # Sort descending by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations: List[Dict[str, Any]] = []
        target_book = self.df.iloc[target_idx].to_dict()

        for idx, score in sim_scores:
            if exclude_self and idx == target_idx:
                continue

            book_row = self.df.iloc[idx].to_dict()
            
            # Find common significant terms between target and recommended book for explainability
            common_terms = self._extract_common_terms(target_idx, idx)

            recommendations.append({
                "book_id": book_row["book_id"],
                "title": book_row["title"],
                "author": book_row["author"],
                "genre": book_row["genre"],
                "description": book_row["description"],
                "publication_year": int(book_row["publication_year"]),
                "rating": float(book_row["rating"]),
                "reading_level": book_row["reading_level"],
                "similarity_score": round(float(score), 4),
                "common_terms": common_terms,
                "explanation": (
                    f"Recommended based on {round(score * 100, 1)}% textual & thematic similarity "
                    f"with '{target_book['title']}'."
                ),
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    def recommend_by_query(
        self,
        query_text: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend books matching a free-text user search query using TF-IDF projection.

        Args:
            query_text: User search phrase or prompt (e.g. "machine learning neural network").
            top_n: Number of recommendations.

        Returns:
            List of recommendation dictionaries.
        """
        if not query_text or not query_text.strip():
            return []

        # Transform query into TF-IDF vector
        query_vec = self.vectorizer.transform([query_text.strip()])

        # Calculate cosine similarity against all books
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort descending
        ranked_indices = np.argsort(sim_scores)[::-1]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(sim_scores[idx])
            # Even if score is low, include until top_n if positive
            if score <= 0.0001 and len(results) > 0:
                break

            book_row = self.df.iloc[idx].to_dict()
            results.append({
                "book_id": book_row["book_id"],
                "title": book_row["title"],
                "author": book_row["author"],
                "genre": book_row["genre"],
                "description": book_row["description"],
                "publication_year": int(book_row["publication_year"]),
                "rating": float(book_row["rating"]),
                "reading_level": book_row["reading_level"],
                "similarity_score": round(score, 4),
                "explanation": (
                    f"Matches your query keywords with {round(score * 100, 1)}% TF-IDF vector cosine match."
                ),
            })

            if len(results) >= top_n:
                break

        return results

    def _extract_common_terms(self, idx1: int, idx2: int, max_terms: int = 4) -> List[str]:
        """Extract highest-weighted overlapping terms between two book vectors."""
        vec1 = self.tfidf_matrix[idx1].toarray().flatten()
        vec2 = self.tfidf_matrix[idx2].toarray().flatten()
        product = vec1 * vec2
        top_term_indices = np.argsort(product)[::-1]
        
        feature_names = self.vectorizer.get_feature_names_out()
        terms: List[str] = []
        for term_idx in top_term_indices:
            if product[term_idx] > 0:
                terms.append(feature_names[term_idx])
            if len(terms) >= max_terms:
                break
        return terms
