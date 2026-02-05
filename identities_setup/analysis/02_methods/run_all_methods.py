"""
Run All Methods Comparison
==========================
Compares four text analysis approaches:
1. DFM + MDS (quanteda-style bag-of-words)
2. Sentence Transformer + MDS (neural embeddings)
3. STM (structural topic model, approximated with NMF)
4. BERTopic (neural topic model)

Usage:
    python run_all_methods.py --data path/to/data.csv --text_col ingroup_lemma

Output:
    - results/comparison_summary.csv
    - results/figures/method_comparison.png
    - docs/methods_comparison.md
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for all methods."""
    RANDOM_STATE = 42
    N_DIMS = 3  # MDS dimensions
    N_TOPICS = 10  # Topic models
    MIN_TOPIC_SIZE = 15  # BERTopic
    EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# =============================================================================
# METHOD 1: DFM + MDS (Bag-of-Words)
# =============================================================================

def run_dfm_mds(texts, config=Config()):
    """
    Traditional document-feature matrix + MDS approach.
    This is what quanteda + cmdscale would do in R.
    """
    print("\n" + "="*60)
    print("METHOD 1: DFM + MDS (Bag-of-Words)")
    print("="*60)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import MDS

    results = {'method': 'DFM_MDS', 'success': False}

    try:
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=2000)
        dtm = vectorizer.fit_transform(texts)

        results['n_features'] = dtm.shape[1]
        results['sparsity'] = 1 - (dtm.nnz / (dtm.shape[0] * dtm.shape[1]))

        print(f"  Documents: {dtm.shape[0]}")
        print(f"  Features: {dtm.shape[1]}")
        print(f"  Sparsity: {results['sparsity']:.1%}")

        # Compute cosine distances
        dist_matrix = pairwise_distances(dtm, metric='cosine')

        # Analyze distance distribution
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        results['mean_distance'] = upper_tri.mean()
        results['no_overlap_rate'] = (upper_tri > 0.99).mean()

        print(f"  Mean distance: {results['mean_distance']:.3f}")
        print(f"  No-overlap pairs: {results['no_overlap_rate']:.1%}")

        # Run MDS
        mds = MDS(n_components=config.N_DIMS, dissimilarity='precomputed',
                  random_state=config.RANDOM_STATE, n_jobs=-1, max_iter=300)
        coords = mds.fit_transform(dist_matrix)

        results['coords'] = coords
        results['stress'] = mds.stress_
        results['success'] = True

        print(f"  MDS stress: {results['stress']:.4f}")

        # Quality assessment
        if results['no_overlap_rate'] > 0.5:
            results['quality'] = 'LOW'
            results['quality_note'] = 'Most document pairs have no word overlap'
        elif results['no_overlap_rate'] > 0.3:
            results['quality'] = 'MEDIUM'
            results['quality_note'] = 'Many document pairs have little overlap'
        else:
            results['quality'] = 'HIGH'
            results['quality_note'] = 'Reasonable word overlap'

        print(f"  Quality: {results['quality']} - {results['quality_note']}")

    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")

    return results

# =============================================================================
# METHOD 2: Sentence Transformer + MDS
# =============================================================================

def run_st_mds(texts, config=Config()):
    """
    Sentence transformer embeddings + MDS approach.
    Uses multilingual model for semantic similarity.
    """
    print("\n" + "="*60)
    print("METHOD 2: Sentence Transformer + MDS")
    print("="*60)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import MDS

    results = {'method': 'ST_MDS', 'success': False}

    try:
        # Load model
        print(f"  Loading model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Encode texts
        print("  Encoding texts...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        results['embedding_dim'] = embeddings.shape[1]
        print(f"  Embedding dimension: {results['embedding_dim']}")

        # Compute cosine distances
        dist_matrix = pairwise_distances(embeddings, metric='cosine')

        # Analyze distance distribution
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        results['mean_distance'] = upper_tri.mean()
        results['similar_pairs_rate'] = (upper_tri < 0.3).mean()

        print(f"  Mean distance: {results['mean_distance']:.3f}")
        print(f"  Similar pairs (<0.3): {results['similar_pairs_rate']:.1%}")

        # Run MDS
        mds = MDS(n_components=config.N_DIMS, dissimilarity='precomputed',
                  random_state=config.RANDOM_STATE, n_jobs=-1, max_iter=500)
        coords = mds.fit_transform(dist_matrix)

        results['coords'] = coords
        results['embeddings'] = embeddings
        results['stress'] = mds.stress_
        results['success'] = True

        print(f"  MDS stress: {results['stress']:.4f}")

        # Quality assessment
        if results['stress'] < 0.15:
            results['quality'] = 'HIGH'
            results['quality_note'] = 'Low stress, good representation'
        elif results['stress'] < 0.25:
            results['quality'] = 'MEDIUM'
            results['quality_note'] = 'Moderate stress'
        else:
            results['quality'] = 'LOW'
            results['quality_note'] = 'High stress, poor fit'

        print(f"  Quality: {results['quality']} - {results['quality_note']}")

    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")

    return results

# =============================================================================
# METHOD 3: STM (Approximated with NMF)
# =============================================================================

def run_stm(texts, config=Config()):
    """
    Structural Topic Model (approximated with NMF for Python).
    True STM requires R; this captures the bag-of-words topic approach.
    """
    print("\n" + "="*60)
    print("METHOD 3: STM-style Topic Model (NMF)")
    print("="*60)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from collections import Counter

    results = {'method': 'STM', 'success': False}

    try:
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        dtm = vectorizer.fit_transform(texts)

        print(f"  Documents: {dtm.shape[0]}")
        print(f"  Features: {dtm.shape[1]}")

        # Fit NMF
        nmf = NMF(n_components=config.N_TOPICS, random_state=config.RANDOM_STATE,
                  max_iter=500)
        doc_topics = nmf.fit_transform(dtm)

        # Get top words per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        print(f"\n  Topics found: {config.N_TOPICS}")
        for i, topic in enumerate(nmf.components_):
            top_words = [feature_names[j] for j in topic.argsort()[:-8:-1]]
            topics.append(top_words)
            print(f"    Topic {i+1}: {', '.join(top_words)}")

        results['topics'] = topics
        results['doc_topics'] = doc_topics

        # Topic distribution
        topic_assignments = doc_topics.argmax(axis=1)
        topic_counts = Counter(topic_assignments)
        results['topic_distribution'] = dict(topic_counts)

        # Quality metrics
        max_confidence = doc_topics.max(axis=1).mean()
        results['mean_confidence'] = max_confidence

        # Check topic overlap (top words shared)
        all_top_words = [set(t[:5]) for t in topics]
        overlaps = []
        for i in range(len(all_top_words)):
            for j in range(i+1, len(all_top_words)):
                overlaps.append(len(all_top_words[i] & all_top_words[j]))
        results['mean_topic_overlap'] = np.mean(overlaps)

        print(f"\n  Mean confidence: {max_confidence:.3f}")
        print(f"  Mean topic overlap: {results['mean_topic_overlap']:.2f} words")

        results['success'] = True

        # Quality assessment
        if results['mean_topic_overlap'] > 1.5:
            results['quality'] = 'LOW'
            results['quality_note'] = 'Topics share too many words'
        elif max_confidence < 0.3:
            results['quality'] = 'LOW'
            results['quality_note'] = 'Low topic assignment confidence'
        else:
            results['quality'] = 'MEDIUM'
            results['quality_note'] = 'Reasonable topic separation'

        print(f"  Quality: {results['quality']} - {results['quality_note']}")

    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")

    return results

# =============================================================================
# METHOD 4: BERTopic
# =============================================================================

def run_bertopic(texts, config=Config()):
    """
    BERTopic neural topic model.
    """
    print("\n" + "="*60)
    print("METHOD 4: BERTopic")
    print("="*60)

    results = {'method': 'BERTopic', 'success': False}

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        print(f"  Loading model: {config.EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Configure BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=config.MIN_TOPIC_SIZE,
            nr_topics='auto',
            verbose=False
        )

        # Fit model
        print("  Fitting BERTopic...")
        topics, probs = topic_model.fit_transform(texts)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # -1 for outlier topic

        results['n_topics'] = n_topics
        results['topics'] = topics
        results['probs'] = probs
        results['topic_info'] = topic_info

        # Outlier analysis
        outlier_rate = (np.array(topics) == -1).mean()
        results['outlier_rate'] = outlier_rate

        print(f"  Topics found: {n_topics}")
        print(f"  Outlier rate: {outlier_rate:.1%}")

        # Show top topics
        print(f"\n  Top topics:")
        for _, row in topic_info.head(6).iterrows():
            if row['Topic'] != -1:
                name = row['Name'][:50] if len(row['Name']) > 50 else row['Name']
                print(f"    Topic {row['Topic']}: {row['Count']} docs - {name}")

        results['success'] = True

        # Quality assessment
        if outlier_rate > 0.4:
            results['quality'] = 'LOW'
            results['quality_note'] = f'Too many outliers ({outlier_rate:.0%})'
        elif n_topics > 50:
            results['quality'] = 'LOW'
            results['quality_note'] = f'Too many micro-topics ({n_topics})'
        elif n_topics < 3:
            results['quality'] = 'LOW'
            results['quality_note'] = 'Too few topics found'
        else:
            results['quality'] = 'MEDIUM'
            results['quality_note'] = f'{n_topics} topics, {outlier_rate:.0%} outliers'

        print(f"  Quality: {results['quality']} - {results['quality_note']}")

    except Exception as e:
        results['error'] = str(e)
        print(f"  ERROR: {e}")

    return results

# =============================================================================
# COMPARISON & REPORTING
# =============================================================================

def create_comparison_summary(results_list, texts, output_dir):
    """Create summary table and visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    # Summary table
    summary = []
    for r in results_list:
        row = {
            'Method': r['method'],
            'Success': r['success'],
            'Quality': r.get('quality', 'N/A'),
            'Note': r.get('quality_note', r.get('error', ''))[:50]
        }

        if r['method'] in ['DFM_MDS', 'ST_MDS']:
            row['Stress'] = f"{r.get('stress', 'N/A'):.4f}" if r.get('stress') else 'N/A'
        if r['method'] == 'DFM_MDS':
            row['No-overlap %'] = f"{r.get('no_overlap_rate', 0):.1%}"
        if r['method'] == 'BERTopic':
            row['Topics'] = r.get('n_topics', 'N/A')
            row['Outliers'] = f"{r.get('outlier_rate', 0):.1%}"

        summary.append(row)

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)

    # Create visualization
    create_comparison_figure(results_list, texts, output_dir)

    # Create markdown report
    create_markdown_report(results_list, texts, output_dir)

    return summary_df

def create_comparison_figure(results_list, texts, output_dir):
    """Create side-by-side MDS comparison plot."""
    import matplotlib.pyplot as plt

    mds_results = [r for r in results_list if 'coords' in r and r['success']]

    if len(mds_results) < 2:
        print("  Not enough MDS results for comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, r in enumerate(mds_results[:2]):
        coords = r['coords']
        ax = axes[i]

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            alpha=0.4, s=15, c='steelblue', edgecolors='none')

        ax.set_title(f"{r['method']}\nQuality: {r['quality']}", fontsize=12)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        # Add stress info
        ax.text(0.02, 0.98, f"Stress: {r['stress']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top')

    plt.suptitle('MDS Comparison: Bag-of-Words vs Sentence Transformer', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'mds_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/mds_comparison.png")

def create_markdown_report(results_list, texts, output_dir):
    """Create detailed markdown report for coauthors."""

    report = """# Methods Comparison Report

## Overview

This report compares four text analysis approaches for analyzing open-ended survey responses about political identity.

**Data:** {n_docs} responses, median length ~6 words

## Summary Table

| Method | Quality | Key Metric | Notes |
|--------|---------|------------|-------|
""".format(n_docs=len(texts))

    for r in results_list:
        quality = r.get('quality', 'N/A')
        note = r.get('quality_note', r.get('error', 'N/A'))[:40]

        if r['method'] == 'DFM_MDS':
            metric = f"No-overlap: {r.get('no_overlap_rate', 0):.0%}"
        elif r['method'] == 'ST_MDS':
            metric = f"Stress: {r.get('stress', 'N/A'):.4f}"
        elif r['method'] == 'BERTopic':
            metric = f"Topics: {r.get('n_topics', 'N/A')}, Outliers: {r.get('outlier_rate', 0):.0%}"
        else:
            metric = f"Confidence: {r.get('mean_confidence', 'N/A'):.2f}"

        report += f"| {r['method']} | {quality} | {metric} | {note} |\n"

    report += """
## Detailed Results

"""

    for r in results_list:
        report += f"### {r['method']}\n\n"

        if not r['success']:
            report += f"**Error:** {r.get('error', 'Unknown error')}\n\n"
            continue

        report += f"**Quality:** {r.get('quality', 'N/A')}\n\n"
        report += f"**Notes:** {r.get('quality_note', 'N/A')}\n\n"

        if r['method'] == 'DFM_MDS':
            report += f"""
- Sparsity: {r.get('sparsity', 0):.1%}
- No-overlap pairs: {r.get('no_overlap_rate', 0):.1%}
- MDS Stress: {r.get('stress', 'N/A'):.4f}

**Interpretation:** {"High no-overlap rate indicates most document pairs share no words, making BOW distances unreliable." if r.get('no_overlap_rate', 0) > 0.3 else "Reasonable word overlap for BOW approach."}

"""
        elif r['method'] == 'ST_MDS':
            report += f"""
- Embedding dimension: {r.get('embedding_dim', 384)}
- Mean distance: {r.get('mean_distance', 'N/A'):.3f}
- Similar pairs: {r.get('similar_pairs_rate', 0):.1%}
- MDS Stress: {r.get('stress', 'N/A'):.4f}

**Interpretation:** Sentence transformer captures semantic similarity regardless of vocabulary overlap.

"""
        elif r['method'] == 'STM':
            report += f"""
- Topics: {len(r.get('topics', []))}
- Mean confidence: {r.get('mean_confidence', 'N/A'):.3f}
- Topic word overlap: {r.get('mean_topic_overlap', 'N/A'):.2f}

**Top Topics:**
"""
            for i, topic in enumerate(r.get('topics', [])[:5]):
                report += f"- Topic {i+1}: {', '.join(topic[:5])}\n"
            report += "\n"

        elif r['method'] == 'BERTopic':
            report += f"""
- Topics found: {r.get('n_topics', 'N/A')}
- Outlier rate: {r.get('outlier_rate', 0):.1%}

**Interpretation:** {"High outlier rate suggests texts are too short/diverse for clustering." if r.get('outlier_rate', 0) > 0.3 else "Reasonable clustering achieved."}

"""

    report += """
## Recommendation

Based on this comparison:

1. **For dimensionality reduction:** Sentence Transformer + MDS produces more interpretable results
   because it captures semantic similarity rather than vocabulary overlap.

2. **For topic modeling:** Neither STM nor BERTopic performs optimally on short texts (~6 words).
   Consider using MDS dimensions rather than discrete topics.

## References

- Lin (2025) - Cross-encoders for short texts, AJPS
- Licht (2023) - Multilingual sentence embeddings, Political Analysis
- Hobbs & Green (2025) - Attitudes vs topics, Political Analysis
"""

    with open(output_dir / 'methods_comparison.md', 'w') as f:
        f.write(report)

    print(f"  Saved: {output_dir}/methods_comparison.md")

# =============================================================================
# MAIN
# =============================================================================

def main(data_path, text_col, output_dir='results'):
    """Run complete methods comparison."""
    print("="*60)
    print("METHODS COMPARISON")
    print("="*60)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Prepare texts
    texts = df[text_col].dropna().astype(str).tolist()
    texts = [t.strip() for t in texts if len(t.strip()) > 3]

    print(f"Documents: {len(texts)}")
    print(f"Median words: {np.median([len(t.split()) for t in texts]):.1f}")

    # Run all methods
    config = Config()
    results = []

    results.append(run_dfm_mds(texts, config))
    results.append(run_st_mds(texts, config))
    results.append(run_stm(texts, config))
    results.append(run_bertopic(texts, config))

    # Create comparison
    summary = create_comparison_summary(results, texts, output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print(f"Results saved to: {output_dir}/")
    print("="*60)

    return results, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV file')
    parser.add_argument('--text_col', required=True, help='Column with text')
    parser.add_argument('--output', default='results', help='Output directory')

    args = parser.parse_args()
    main(args.data, args.text_col, args.output)
