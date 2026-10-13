# Algorithm & Mathematical Methodology Guide

**Project Title:** AI-Based Intelligent Book Recommendation System Using Content-Based Filtering and Rule-Based Reasoning  
**Subject:** Artificial Intelligence (Undergraduate Mini-Project)

---

## 1. Overview of Hybrid Architecture

This recommendation system employs a **hybrid recommendation architecture** that synthesizes two complementary AI paradigms:
1. **Content-Based Filtering (Inductive / Vector Space Model):** Computes lexical and thematic proximity using **TF-IDF (Term Frequency - Inverse Document Frequency)** and **Cosine Similarity**.
2. **Rule-Based Reasoning (Deductive / Production System):** Evaluates user preferences against structured book attributes through deterministic **IF-THEN production rules**.
3. **Score Fusion:** Integrates both scores using a normalized convex weighted combination.

```
+-------------------------------------------------------------+
|                        User Input                           |
|  - Preference Criteria (Genre, Reading Level, Rating, Era)  |
|  - Optional Seed Book or Search Query                       |
+------------------------------+------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
| Content-Based Engine  |             |  Rule-Based Engine    |
| - TF-IDF Vectorizer   |             |  - Genre Rule         |
| - Cosine Similarity   |             |  - Reading Level Rule |
|                       |             |  - Rating Quality     |
|                       |             |  - Era & Topic Rules  |
+-----------+-----------+             +-----------+-----------+
            |                                     |
            | Sim(u, v) in [0, 1]                 | Rule(u) in [0, 1]
            |                                     |
            +------------------+------------------+
                               |
                               v
               +-------------------------------+
               |    Weighted Score Fusion      |
               |  Hybrid = w_c*Sim + w_r*Rule  |
               +---------------+---------------+
                               |
                               v
               +-------------------------------+
               |  Ranked Recommendations &     |
               |  Explainability Audit Trail   |
               +-------------------------------+
```

---

## 2. Content-Based Filtering: Mathematics

### 2.1 Term Frequency - Inverse Document Frequency (TF-IDF)
Given a corpus of $N$ books, each book's description and metadata soup is transformed into a high-dimensional vector in term space.

1. **Term Frequency (TF):** Measures how frequently term $t$ appears in document $d$:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
   Where $f_{t, d}$ is the raw count of term $t$ in document $d$.

2. **Inverse Document Frequency (IDF):** Quantifies how rare or informative term $t$ is across the entire corpus of $N$ documents:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + |\ \{d \in D : t \in d\}\ |}\right) + 1$$
   *Note: Smooth IDF formulation with Laplacian smoothing is used to prevent division by zero.*

3. **TF-IDF Weight:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

4. **L2 Normalization:**
   Vectors are normalized to unit Euclidean length:
   $$\vec{v}_d = \frac{\vec{w}_d}{\|\vec{w}_d\|_2}$$

### 2.2 Cosine Similarity
To evaluate similarity between two book vectors $\vec{A}$ and $\vec{B}$, we calculate the cosine of the angle $\theta$ between them:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{m} A_i B_i}{\sqrt{\sum_{i=1}^{m} A_i^2} \sqrt{\sum_{i=1}^{m} B_i^2}}$$

- **Range:** Since TF-IDF weights are non-negative, $\text{Cosine Similarity} \in [0.0, 1.0]$.
- **$1.0$** indicates identical directional orientation (maximum textual similarity).
- **$0.0$** indicates orthogonal vectors (zero shared tokens).

---

## 3. Rule-Based Reasoning: IF-THEN Production System

The rule engine operates as a declarative forward reasoning system. It executes five distinct rule categories, accumulating score points that are strictly normalized.

### Rule Set Specification:

1. **Rule 1: Genre Preference Rule**
   - **Condition:** $\text{IF } \text{book}.\text{genre} \in \text{user}.\text{preferred\_genres}$
   - **Action:** Add $+0.35$ to raw score; log match in audit trail.
   - **Fallback:** If user specified no genre, award neutral $+0.175$.

2. **Rule 2: Reading Level Rule**
   - **Condition (Exact):** $\text{IF } \text{book}.\text{reading\_level} == \text{user}.\text{reading\_level}$
   - **Action:** Add $+0.25$ to raw score.
   - **Condition (Adjacent):** $\text{IF } \text{abs}(\text{level\_ordinal\_diff}) == 1$
   - **Action:** Add $+0.10$ to raw score (partial compatibility credit).

3. **Rule 3: Quality / Rating Threshold Rule**
   - **Condition:** $\text{IF } \text{book}.\text{rating} \ge \text{user}.\text{min\_rating}$
   - **Action:** Add $\left(\frac{\text{book}.\text{rating}}{5.0}\right) \times 0.20$ to raw score.

4. **Rule 4: Publication Era Rule**
   - **Condition:**
     - $\text{Classic}: \text{year} < 1980 \implies +0.10$
     - $\text{Modern}: \text{year} \ge 1980 \implies +0.10$
     - $\text{Contemporary}: \text{year} \ge 2010 \implies +0.10$

5. **Rule 5: Topic Keyword Matching Rule**
   - **Condition:** $\text{IF } \text{keywords} \cap (\text{book}.\text{title} \cup \text{book}.\text{description}) \ne \emptyset$
   - **Action:** Add up to $+0.10$ proportional to keyword overlap ratio.

### Normalization:
$$\text{Rule Score}_{\text{norm}} = \frac{\sum \text{Awarded Points}}{\sum \text{Max Possible Points}} = \frac{\text{raw\_score}}{1.00} \in [0.0, 1.0]$$

---

## 4. Hybrid Score Fusion & Ranking

To rank candidates, the system applies convex score combination:

$$\text{Hybrid Score}(b) = (w_{\text{content}} \times \text{Sim}_{\text{content}}(b)) + (w_{\text{rule}} \times \text{Rule}_{\text{score}}(b))$$

Where:
- $w_{\text{content}} + w_{\text{rule}} = 1.0$ ($w_{\text{content}}, w_{\text{rule}} \ge 0$)
- User can interactively slide the balance between $0\%$ (pure rule-based) and $100\%$ (pure TF-IDF).
- Default configuration: $w_{\text{content}} = 0.50$, $w_{\text{rule}} = 0.50$.

### Tie-Breaking Strategy:
If two books achieve identical hybrid scores:
1. Primary sort key: $\text{Hybrid Score}$ (descending)
2. Secondary sort key: Book rating (descending)
3. Tertiary sort key: Publication year (descending)

---

## 5. Algorithmic Complexity

- **TF-IDF Fitting:** $\mathcal{O}(N \times L)$, where $N$ is catalog size ($111$ books) and $L$ is average description token length.
- **Pairwise Cosine Matrix:** $\mathcal{O}(N^2 \times M)$, where $M$ is feature vocabulary dimension ($2500$ terms). Since $N = 111$, calculation completes in under $30$ milliseconds.
- **Rule Evaluation:** $\mathcal{O}(N \times R)$, where $R$ is the number of rules ($5$). Operates in linear time $\mathcal{O}(N)$.
- **Sorting Top-K:** $\mathcal{O}(N \log N)$ or $\mathcal{O}(N \log K)$ with min-heap.
