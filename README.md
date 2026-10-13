# AI-Based Intelligent Book Recommendation System

An undergraduate Artificial Intelligence mini-project demonstrating **Content-Based Filtering** (TF-IDF vector space modeling + Cosine Similarity) fused with transparent **Rule-Based Reasoning** (declarative IF-THEN production rules).

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Tests](https://img.shields.io/badge/Tests-18%20Passed%20(100%25)-brightgreen)
![License](https://img.shields.io/badge/Academic-Open%20Source-purple)

---

## 📌 Project Highlights
- **Hybrid AI Engine:** Combines statistical text mining with deterministic expert rules.
- **Zero Cold-Start Penalty:** Operates purely on book attributes and explicit user constraints without needing historical user rating matrices.
- **Explainable AI (XAI):** Every single recommendation includes an explainability audit trail showing which rules fired, score breakdowns, and top overlapping TF-IDF terms.
- **Configurable Score Fusion:** Interactive slider allows users to adjust the balance between content similarity ($w_{\text{content}}$) and rule criteria ($w_{\text{rule}}$).
- **Curated Dataset:** 111 distinct books spanning Fiction, Science, Technology, Self-Help, Business, History, Philosophy, and Education in `data/books.csv`.
- **Persistent User Feedback:** Thumbs Up / Down ratings saved locally to `data/feedback.csv` with optional empirical reranking.
- **100% Test Coverage:** 18 automated unit tests covering all algorithmic edge cases with `pytest`.
- **100% Local & Free:** No external paid APIs, no API keys, and no database servers required.

---

## 🏗️ Project Architecture

```
d:\Projects\AIL_mini\
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore specification
├── README.md                  # Project documentation & run guide
├── data/
│   ├── books.csv              # 111 curated books across 8 genres
│   └── feedback.csv           # Persistent user feedback store
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # CSV parsing, schema validation & sanitization
│   ├── recommender.py         # TF-IDF vectorizer & Cosine Similarity engine
│   ├── rule_engine.py         # IF-THEN rule system & hybrid score fusion
│   ├── feedback.py            # User feedback logging & boost calculations
│   └── analytics.py           # Matplotlib data visualizers
├── tests/
│   ├── __init__.py            # Test package initialization
│   ├── test_data_loader.py    # Schema & sanitization unit tests
│   ├── test_recommender.py    # TF-IDF & similarity unit tests
│   └── test_rule_engine.py    # Rule evaluation & hybrid fusion tests
└── docs/
    ├── project_report.md      # Full academic project report with Mermaid diagrams
    ├── algorithm.md           # Mathematical formulation & pseudocode guide
    └── viva_questions.md      # 25 in-depth viva questions & answers
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
Ensure Python 3.10 or higher is installed on your system.

### 2. Install Dependencies
Open your command terminal (PowerShell, CMD, or Bash) in the project root directory and install dependencies:

```bash
pip install -r requirements.txt
```

*(Or individual packages: `pip install streamlit pandas numpy scikit-learn matplotlib pytest`)*

### 3. Run the Automated Test Suite
Verify that all algorithms and data layers are operating correctly:

```bash
python -m pytest -v
```

Expected output:
```
============================= 18 passed in 4.58s ==============================
```

### 4. Launch the Web Application
Start the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🧭 Application Walkthrough

| Module | Purpose & Features |
| :--- | :--- |
| **🏠 Dashboard & Overview** | High-level summary, catalog KPI cards, genre breakdown, and system features. |
| **⚡ Hybrid Recommender** | Select genres, target reading levels (Beginner, Intermediate, Advanced), publication era, and adjust the Content vs. Rule slider to get personalized, explainable recommendations. |
| **🔍 Book-to-Book Similarity** | Pick any reference book to find top 5 or 10 most similar titles using pure TF-IDF cosine similarity. |
| **📖 Catalog Explorer** | Search across 111 books by keyword, author, or title, with multi-level filtering. |
| **📊 Analytics & Insights** | Interactive charts of genre distributions, rating distributions, and user feedback metrics. |
| **🎓 Viva & Algorithm Guide** | Embedded academic cheat sheet with system formulas and quick viva answers. |

---

## 🧮 Mathematical Formulation

### 1. TF-IDF and Cosine Similarity
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

### 2. Rule Evaluation
Declarative forward evaluation over 5 distinct rules:
- **Rule 1 (Genre):** $+0.35$ on match
- **Rule 2 (Reading Level):** $+0.25$ on exact match, $+0.10$ on adjacent difficulty
- **Rule 3 (Rating Quality):** $\left(\frac{\text{rating}}{5.0}\right) \times 0.20$
- **Rule 4 (Publication Era):** $+0.10$ on era alignment
- **Rule 5 (Topic Keywords):** $+0.10$ on keyword overlap

### 3. Hybrid Score Fusion
$$\text{Hybrid Score}(b) = w_{\text{content}} \times \text{Sim}_{\text{content}}(b) + w_{\text{rule}} \times \text{Rule}_{\text{score}}(b)$$
Where $w_{\text{content}} + w_{\text{rule}} = 1.0$ and both scores are normalized to $[0.0, 1.0]$.

---

## 📚 Academic Documentation Included
- **[Full Project Report](docs/project_report.md):** Complete academic report with editable student header, abstract, problem statement, Mermaid system architecture, pseudocode, and references.
- **[Algorithm Guide](docs/algorithm.md):** Mathematical formulas and complexity analysis.
- **[Viva Questions & Answers](docs/viva_questions.md):** 25 likely viva questions with model answers.

---

## 🔧 Troubleshooting

- **Streamlit Port In Use:** If port 8501 is occupied, run:  
  `streamlit run app.py --server.port 8502`
- **Missing Dataset:** If `data/books.csv` is accidentally deleted, check `src/data_loader.py` for schema requirements or restore from git.
- **Stale Cache:** Click the **"🔄 Refresh Catalog Cache"** button in the sidebar to invalidate Streamlit's cache.
