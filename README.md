# KHDL Lab02 - LaTeX Reference Matching Pipeline

**Student ID:** 23120260

## Overview

This project provides a complete pipeline for:
1. **Hierarchical Parsing and Standardization** of LaTeX documents
2. **Reference Matching** between BibTeX citations and arXiv references

---

## Project Structure

```
KHDL_Lab02_v2/
├── src/
│   ├── parsing/                    # Hierarchical Parsing (.py modules)
│   │   ├── __init__.py
│   │   ├── parsing.py              # Main pipeline controller
│   │   ├── file_gatherer.py        # Multi-file gathering (traces \input{})
│   │   ├── latex_parser.py         # LaTeX parsing + BibTeX conversion
│   │   ├── standardizer.py         # Text standardization
│   │   ├── hierarchy_builder.py    # Hierarchical structure + sentence splitting
│   │   └── deduplicator.py         # Cross-version deduplication
│   │
│   └── matching/                   # Reference Matching (.ipynb notebooks)
│       ├── __init__.py             # Data structures (BibEntry, RefEntry, TextCleaner)
│       ├── 00_manual_labeling.ipynb # Manual ground truth (5 pubs, 52 labels)
│       ├── 01_data_cleaning.ipynb  # Data extraction + auto-labeling (10%)
│       ├── 02_feature_engineering.ipynb  # 12 matching features
│       ├── 03_model_training.ipynb # Publication-level train/valid/test split
│       └── 04_evaluation.ipynb     # MRR@5 evaluation + pred.json generation
│
├── 23120260/                       # Data directory
│   └── 2411-XXXXX/                 # Publication folders
│       ├── tex/                    # LaTeX source files
│       ├── references.json         # Candidate arXiv references
│       ├── metadata.json           # Publication metadata
│       └── pred.json               # Predictions (generated)
│
├── output/                         # Parsing output
│   └── 23120260/
│       └── 2411-XXXXX/
│           └── <pub_id>.json       # Hierarchical document structure
│
└── README.md
```

---

## Part 1: Hierarchical Parsing and Standardization (2.1)

### Modules

| Module | Purpose |
|--------|---------|
| `parsing.py` | Main pipeline controller |
| `file_gatherer.py` | Traces `\input{}` and `\include{}` commands |
| `latex_parser.py` | Parses LaTeX, converts bibitem to BibTeX |
| `standardizer.py` | Normalizes math, cleans LaTeX commands |
| `hierarchy_builder.py` | Builds hierarchy with sentence leaf nodes |
| `deduplicator.py` | Cross-version reference deduplication |

### Features Implemented

| Requirement | Implementation |
|-------------|----------------|
| **2.1.1 Multi-file Gathering** | `_trace_includes()` follows actual `\input{}` paths |
| **2.1.2 Hierarchy - Sections** | Detects `\section`, `\subsection`, `\subsubsection` |
| **2.1.2 Hierarchy - Leaf Nodes** | Sentences, block formulas, figures (not raw paragraphs) |
| **2.1.2 Exclude References** | `_is_reference_section()` filters out bibliography |
| **2.1.3 Standardization** | LaTeX cleanup, math normalization, `[cite]`/`[ref]` |
| **2.1.3 BibTeX Conversion** | `_parse_bibitem_to_bibtex()` extracts fields |
| **2.1.3 Deduplication** | Hash-based content dedup, cross-version reference dedup |

### Output Format

```json
{
  "elements": {
    "document_0001": "2411-00222",
    "title_0002": "Paper Title Here",
    "section_0003": "Introduction",
    "sentence_0004": "First sentence of the introduction.",
    "sentence_0005": "Second sentence with citation [cite].",
    "equation_0006": "E = mc^2",
    "figure_0007": "Figure caption text"
  },
  "hierarchy": {
    "1": {
      "title_0002": "document_0001",
      "section_0003": "document_0001",
      "sentence_0004": "section_0003",
      "sentence_0005": "section_0003",
      "equation_0006": "document_0001",
      "figure_0007": "document_0001"
    }
  }
}
```

### Usage

```python
from src.parsing import ParsingPipeline

pipeline = ParsingPipeline(base_path="C:/Code/KHDL_Lab02_v2")

# Process single publication
result = pipeline.process_publication("23120260", "2411-00222")
print(f"Nodes: {result.statistics['total_nodes']}")

# Process all publications
results = pipeline.process_all("23120260", limit=1000)
```

```bash
# Run from command line
python -m src.parsing.parsing
```

---

## Part 2: Reference Matching Pipeline (2.2)

### Pipeline Overview

```
00_manual_labeling.ipynb  →  Manual labels (5 pubs, 52+ pairs)
        ↓
01_data_cleaning.ipynb    →  Extract bibs + Auto-label (10% coverage)
        ↓
02_feature_engineering.ipynb  →  Extract 12 matching features
        ↓
03_model_training.ipynb   →  Train model (publication-level split)
        ↓
04_evaluation.ipynb       →  Evaluate MRR@5 + Generate pred.json
```

### 2.2.1 Data Cleaning

Implemented in `01_data_cleaning.ipynb`:
- Lowercasing, stop-word removal
- LaTeX command cleanup
- Author name normalization (extract last names)
- Tokenization

### 2.2.2 Data Labeling

**Manual Labels** (`00_manual_labeling.ipynb`):
- 5 publications manually labeled
- 52 total label pairs (exceeds 20 minimum)
- Publications: 2411-00222, 2411-00223, 2411-00225, 2411-00226, 2411-00227

**Automatic Labels** (`01_data_cleaning.ipynb`):
- `AutoLabeler` class with 3 strategies:
  1. Exact arXiv ID match in bibitem content
  2. High title Jaccard similarity (≥0.7)
  3. First author + year + partial title match
- Covers 10%+ of non-manual data

### 2.2.3 Feature Engineering

12 features implemented in `02_feature_engineering.ipynb`:

| Feature | Description |
|---------|-------------|
| `title_jaccard` | Jaccard similarity of title tokens |
| `title_overlap` | Token overlap ratio |
| `title_edit_dist` | 1 - normalized edit distance |
| `author_overlap` | Author last name overlap |
| `first_author_match` | Binary: first authors match |
| `year_match` | Binary: exact year match |
| `year_diff` | Absolute year difference |
| `arxiv_match` | Binary: arXiv IDs match |
| `arxiv_in_content` | Binary: arXiv ID in raw content |
| `num_matching_authors` | Count of matching authors |
| `title_len_ratio` | Ratio of title lengths |
| `combined_score` | Weighted combination |

### 2.2.4 Data Modeling

**Model:** Logistic regression with gradient descent

**Publication-Level Split:**
| Set | Manual | Auto | Total |
|-----|--------|------|-------|
| Test | 1 (2411-00222) | 1 | 2 pubs |
| Valid | 1 (2411-00223) | 1 | 2 pubs |
| Train | 3 | remaining | rest |

### 2.2.5 Model Evaluation

**Metric:** MRR@5 (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

- $|Q|$ = total references to match
- $rank_i$ = position of correct match in top-5 list (0 if not found)

---

## Part 3: Output Format (3.1.3)

### pred.json Format

```json
{
  "partition": "test",
  "groundtruth": {
    "bibtex_entry_name_1": "arxiv_id_from_references_json"
  },
  "prediction": {
    "bibtex_entry_name_1": ["candidate_id_1", "candidate_id_2", "candidate_id_3"]
  }
}
```

- **partition**: `"train"`, `"valid"`, or `"test"`
- **groundtruth**: BibTeX key → correct arXiv ID
- **prediction**: BibTeX key → ranked list of top-5 candidates

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install numpy jupyter

# 3. Run parsing pipeline
python -m src.parsing.parsing

# 4. Run matching pipeline
cd src/matching
jupyter notebook
# Execute notebooks 00 → 01 → 02 → 03 → 04
```

---

## Requirements

```
Python 3.8+
numpy
jupyter
```

---

## Summary

| Requirement | Status | Location |
|-------------|--------|----------|
| 2.1.1 Multi-file Gathering | ✅ | `file_gatherer.py` |
| 2.1.2 Hierarchy Construction | ✅ | `hierarchy_builder.py` |
| 2.1.2 Sentence Leaf Nodes | ✅ | `_split_into_sentences()` |
| 2.1.2 Exclude References | ✅ | `_is_reference_section()` |
| 2.1.3 Standardization | ✅ | `standardizer.py` |
| 2.1.3 BibTeX Conversion | ✅ | `latex_parser.py` |
| 2.1.3 Deduplication | ✅ | `deduplicator.py` |
| 2.2.1 Data Cleaning | ✅ | `01_data_cleaning.ipynb` |
| 2.2.2 Manual Labels (20+) | ✅ | `00_manual_labeling.ipynb` (52 labels) |
| 2.2.2 Auto Labels (10%) | ✅ | `01_data_cleaning.ipynb` |
| 2.2.3 Feature Engineering | ✅ | `02_feature_engineering.ipynb` (12 features) |
| 2.2.4 Publication-Level Split | ✅ | `03_model_training.ipynb` |
| 2.2.5 MRR@5 Evaluation | ✅ | `04_evaluation.ipynb` |
| 3.1.3 pred.json Format | ✅ | `04_evaluation.ipynb` |

---

**Student ID:** 23120260

