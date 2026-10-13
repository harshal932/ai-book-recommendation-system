# Top 25 AI Mini-Project Viva Questions & In-Depth Answers

**Project Title:** AI-Based Intelligent Book Recommendation System Using Content-Based Filtering and Rule-Based Reasoning  
**Target Course:** B.Tech / B.E. / B.Sc / BCA in Computer Science / Artificial Intelligence  

---

### Q1: What is the main objective of this project?
**Answer:**  
The objective is to design, implement, and evaluate a fully transparent, hybrid book recommendation system. It combines inductive statistical text vectorization (TF-IDF and Cosine Similarity) with deductive knowledge engineering (declarative IF-THEN production rules) to deliver relevant, personalized book suggestions with a complete explainability audit trail.

---

### Q2: What is the difference between Content-Based Filtering and Collaborative Filtering?
**Answer:**  
- **Content-Based Filtering:** Recommends items by matching the attributes/metadata of items (such as book synopsis, genre, author) with a user’s explicit profile or target book. It treats each user independently.
- **Collaborative Filtering:** Recommends items based on the past behavior, preferences, and ratings of a large community of users ("users who liked X also liked Y"). It does not require analyzing item content.

---

### Q3: What is the "Cold-Start Problem" and how does your project address it?
**Answer:**  
The Cold-Start problem occurs in collaborative filtering when a new user joins (no historical ratings) or a new item is added (no community feedback). Collaborative systems cannot recommend in this state.  
Our system resolves the Cold-Start problem because Content-Based and Rule-Based models do not rely on crowd interaction history. They work immediately on day one using item metadata (description, genre, level) and explicit user constraints.

---

### Q4: Why did you use TF-IDF rather than simple Word Count (Bag-of-Words)?
**Answer:**  
Simple word count counts occurrences indiscriminately, biasing results toward common words (e.g., "book", "story", "life", "written") that appear frequently across all books but provide little discriminative power.  
TF-IDF multiplies Term Frequency (local importance) by Inverse Document Frequency (global rarity). Words that appear in almost all books receive near-zero weights, while domain-specific terms (e.g., "quantum", "dystopian", "algorithms") receive high weights.

---

### Q5: Explain the mathematical formula for TF-IDF.
**Answer:**  
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Where:
- $\text{TF}(t, d) = \frac{\text{Count of } t \text{ in document } d}{\text{Total tokens in } d}$
- $\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + \text{Count of documents containing } t}\right) + 1$
where $N$ is the total number of books in the catalog.

---

### Q6: Why do we use Cosine Similarity instead of Euclidean Distance?
**Answer:**  
Euclidean distance measures the geometric distance between point coordinates, which is sensitive to document length (magnitude). A short description and a long, detailed description about the same subject would have large Euclidean distance due to word count disparity.  
Cosine similarity measures the cosine of the angle between vectors:
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
Because vectors are normalized, Cosine Similarity evaluates directional thematic orientation independent of document length.

---

### Q7: What is Rule-Based Reasoning in Artificial Intelligence?
**Answer:**  
Rule-Based Reasoning is a classical symbolic AI paradigm where domain knowledge is codified as declarative production rules of the form `IF (condition) THEN (action/conclusion)`. In our system, rules evaluate user criteria such as reading difficulty, genre preference, rating thresholds, and historical era, generating structured match scores and audit trails.

---

### Q8: How are the Content-Based and Rule-Based components combined?
**Answer:**  
They are combined using a normalized convex weighted fusion formula:
$$\text{Hybrid Score} = w_{\text{content}} \times S_{\text{content}} + w_{\text{rule}} \times S_{\text{rule}}$$
where $w_{\text{content}} + w_{\text{rule}} = 1.0$. Both component scores are strictly normalized to the range $[0.0, 1.0]$. The user can dynamically adjust the weighting slider in the user interface.

---

### Q9: Does your system claim to implement Forward or Backward Chaining?
**Answer:**  
No. Our engine uses direct, declarative production rule evaluation with scoring accumulators and an audit trail. We do not make unsupported claims of full inference engine chaining, ensuring academic honesty and technical accuracy.

---

### Q10: How does your system exclude the query book from its own recommendations?
**Answer:**  
When generating similar books for a target book $B_k$, the engine explicitly filters out index $k$ during sorting (`if b_id == seed_book_id: continue`). This guarantees that a book never recommends itself with a trivial similarity score of $1.0$.

---

### Q11: How do you handle missing values or corrupted records in the dataset?
**Answer:**  
In `src/data_loader.py`, we implement automated schema verification and data sanitization:
- Missing descriptions are imputed with empty strings.
- Missing titles fallback to `"Untitled Book (<book_id>)"`.
- Missing or non-numeric ratings fallback to catalog median ($4.0$) and are clamped to $[1.0, 5.0]$.
- Invalid reading levels default to `"Intermediate"`.
- Duplicate `book_id` records are deduplicated keeping the first occurrence.

---

### Q12: What does "Explainability in AI" (XAI) mean in the context of this system?
**Answer:**  
Explainability means the user is not presented with opaque predictions. For every recommended book, our system displays:
1. Exact percentage score breakdown (Content % vs. Rule %).
2. Top overlapping TF-IDF terms between the query/seed and the recommended book.
3. A transparent audit trail of all active IF-THEN rules that triggered and points awarded.

---

### Q13: Does the model automatically retrain itself from user feedback?
**Answer:**  
No. Standard TF-IDF vector space models do not update term dictionaries continuously in real time. We record feedback persistently in `data/feedback.csv` and provide an explicit, documented empirical modifier (`compute_feedback_boost`) that adjusts rank weights by up to $\pm 0.05$ based on historical helpfulness ratios.

---

### Q14: What is the vocabulary size and n-gram range used in your TF-IDF vectorizer?
**Answer:**  
The vectorizer uses unigrams and bigrams (`ngram_range=(1, 2)`), English stop words removal, and a maximum feature cap of 2,500 terms. Using bigrams captures contextual phrases like "machine learning" or "data structures" that have distinct meaning beyond single words.

---

### Q15: What is the time complexity of generating recommendations?
**Answer:**  
- Content similarity against precomputed cosine matrix: $\mathcal{O}(1)$ lookup or $\mathcal{O}(N \times M)$ for free-text queries ($M = 2500$ terms).
- Rule evaluation: $\mathcal{O}(N \times R)$ where $R = 5$ rules.
- Ranking top $K$: $\mathcal{O}(N \log N)$.
For our 111-book catalog, execution completes in less than 20 milliseconds.

---

### Q16: What are the primary limitations of Content-Based Filtering?
**Answer:**  
1. **Limited Serendipity (Over-specialization):** Recommends items very similar to what the user already knows; unlikely to discover surprising cross-genre favorites.
2. **Subjective Quality Ignorance:** TF-IDF measures word usage, not literary quality or narrative prose elegance.
3. **Keyword Sensitivity:** Synonyms not captured in the vector vocabulary may cause relevant books to be missed unless latent semantic analysis (LSA/embeddings) is used.

---

### Q17: What are the primary advantages of this hybrid approach?
**Answer:**  
1. Immediate operation with zero cold-start penalty.
2. 100% transparent and explainable.
3. User maintains interactive control over algorithmic weighting.
4. Lightweight, runs on any laptop without GPU or external cloud API keys.

---

### Q18: What libraries are used in this project and why?
**Answer:**  
- **Streamlit:** For a reactive, pure-Python web dashboard without complex frontend overhead.
- **Pandas & NumPy:** For tabular operations, schema validation, and vector manipulations.
- **scikit-learn:** For production-grade `TfidfVectorizer` and `cosine_similarity`.
- **Matplotlib:** For rendering publication-quality catalog distribution charts.
- **pytest:** For automated test suites covering edge cases and algorithms.

---

### Q19: How did you test your algorithms?
**Answer:**  
We created a 18-test automated suite using `pytest` verifying:
- Schema enforcement and missing value handling.
- Self-exclusion of seed books.
- Boundedness of similarity and hybrid scores in $[0.0, 1.0]$.
- Priority of matching genre and reading levels in the rule engine.
- Extreme edge cases (empty search query, unknown book ID, empty user preferences).
All 18 tests execute and pass with 100% success.

---

### Q20: How could this project be extended for a final-year major project?
**Answer:**  
1. Integrate Dense Semantic Embeddings (e.g., Sentence-BERT / MiniLM) instead of sparse TF-IDF.
2. Implement Matrix Factorization (SVD / ALS) to combine Collaborative Filtering once user ratings scale.
3. Add a Knowledge Graph of literary authors, themes, and historical movements.
4. Deploy as a containerized microservice with FastAPI and Docker.

---

### Q21: What is an n-gram in Natural Language Processing?
**Answer:**  
An n-gram is a contiguous sequence of $n$ items from a given sample of text. A 1-gram (unigram) is a single word ("learning"), while a 2-gram (bigram) is a two-word pair ("machine learning"). Using bigrams preserves word co-occurrence semantics.

---

### Q22: What happens if a user submits an empty query or no preference filters?
**Answer:**  
The system handles this gracefully:
- Free-text query: Returns an empty list or prompts the user for input without throwing an exception.
- Rule engine: If `UserPreferences.is_empty()` is True, assigns a neutral baseline score ($0.50$) and documents "Neutral baseline score applied" in the audit trail.

---

### Q23: Why is reading level an important rule attribute in a book recommender?
**Answer:**  
A book can match a user’s topic interest perfectly (e.g., "Algorithms") but be completely mismatched in difficulty (e.g., Cormen's mathematical CLRS textbook vs. Grokking Algorithms illustrated guide). Matching reading difficulty prevents reader frustration.

---

### Q24: What is the significance of the secondary sort key during ranking?
**Answer:**  
When multiple books have equal hybrid scores (tie score), the secondary key sorts by verified book rating, ensuring higher-quality titles take precedence.

---

### Q25: Can this architecture be adapted to other domains like movies or job postings?
**Answer:**  
Yes! The exact same architectural pattern applies:
- **Items:** Movie synopses or Job descriptions (TF-IDF Content Similarity).
- **Rules:** Age rating, genre, duration / Experience level, salary threshold, remote status (Rule Engine).
- **Fusion:** Weighted score combination.
