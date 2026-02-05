# Political Identities in CEE

Comparative analysis of political identity construction in Czech Republic, Hungary, and Lithuania using open-ended survey responses.

## Project Overview

**Research Question:** How do citizens in CEE countries construct their political ingroups and outgroups?

**Data:** ~1,400 respondents per country, open-ended survey questions

**Methods Comparison:**
| Approach | Bag-of-Words | Neural Embeddings |
|----------|--------------|-------------------|
| Topic Model | STM | BERTopic |
| Dimensionality Reduction | DFM + MDS | Sentence Transformer + MDS |

## Repository Structure

```
identities/
├── data/
│   ├── raw/              # Original survey data (don't edit)
│   └── processed/        # Cleaned/lemmatized data
│
├── analysis/
│   ├── 01_preprocessing/ # Data cleaning, lemmatization
│   ├── 02_methods/       # STM, BERTopic, DFM-MDS, ST-MDS
│   └── 03_comparison/    # Methods comparison
│
├── results/
│   ├── figures/          # All plots
│   └── tables/           # All tables
│
├── docs/
│   └── methods_comparison.md  # Summary for coauthors
│
└── notebooks/
    └── full_analysis.ipynb    # Master Colab notebook
```

## Quick Start

### Option 1: Run in Google Colab
1. Open `notebooks/full_analysis.ipynb` in Colab
2. Upload your data files when prompted
3. Run all cells

### Option 2: Run locally
```bash
pip install -r requirements.txt
python analysis/02_methods/run_all_methods.py
```

## Methods

### 1. Sentence Transformer + MDS (Primary)
- Uses `paraphrase-multilingual-MiniLM-L12-v2`
- Captures semantic similarity regardless of vocabulary overlap
- Validated for political texts (Licht 2023)

### 2. DFM + MDS (Comparison)
- Traditional bag-of-words approach
- Expected to struggle with short texts (~6 words median)

### 3. STM (Comparison)
- Structural Topic Models
- Standard in political science

### 4. BERTopic (Comparison)
- Neural topic model
- May produce too many clusters for short texts

## Key References

- Lin (2025) - Cross-encoders for short political texts (AJPS)
- Licht (2023) - Multilingual sentence embeddings (Political Analysis)
- Hobbs & Green (2025) - Attitudes vs topics in surveys (Political Analysis)
- Reimers & Gurevych (2019) - Sentence-BERT

## Authors

- [Your names here]

## Data Availability

[Note about data access]
