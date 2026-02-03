"""
Combined Ingroup-Outgroup MDS Analysis
=======================================
This script runs MDS on ingroup and outgroup descriptions together
to test whether they occupy opposite poles in a shared semantic space.

Run in Google Colab with GPU for faster encoding.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# These imports require installation in Colab:
# !pip install sentence-transformers scikit-learn

def run_combined_mds(data_path='clean_data_with_improved_lemmas.csv'):
    """
    Run MDS on combined ingroup and outgroup texts.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    print("Loading data...")
    df = pd.read_csv(data_path)

    # Get text columns (adjust column names as needed)
    ingroup_col = 'ingroup_lemma' if 'ingroup_lemma' in df.columns else 'ingroup_text'
    outgroup_col = 'outgroup_lemma' if 'outgroup_lemma' in df.columns else 'outgroup_text'

    print(f"Using columns: {ingroup_col}, {outgroup_col}")

    # Clean and prepare texts
    ingroup_texts = df[ingroup_col].dropna().astype(str).tolist()
    outgroup_texts = df[outgroup_col].dropna().astype(str).tolist()

    # Filter empty strings
    ingroup_texts = [t for t in ingroup_texts if len(t.strip()) > 3]
    outgroup_texts = [t for t in outgroup_texts if len(t.strip()) > 3]

    print(f"Ingroup responses: {len(ingroup_texts)}")
    print(f"Outgroup responses: {len(outgroup_texts)}")

    # Combine texts with labels
    combined_texts = ingroup_texts + outgroup_texts
    labels = ['ingroup'] * len(ingroup_texts) + ['outgroup'] * len(outgroup_texts)

    print(f"\nTotal combined: {len(combined_texts)}")

    # Load sentence transformer
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Encode all texts
    print("Encoding texts (this may take a few minutes)...")
    embeddings = model.encode(combined_texts, show_progress_bar=True, batch_size=64)

    # Compute pairwise distances
    print("\nComputing pairwise cosine distances...")
    dist_matrix = pairwise_distances(embeddings, metric='cosine')

    # Run MDS
    print("Running MDS (3 dimensions)...")
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42,
              n_jobs=-1, max_iter=500, verbose=1)
    coords = mds.fit_transform(dist_matrix)

    print(f"MDS stress: {mds.stress_:.2f}")

    # Create results dataframe
    results = pd.DataFrame({
        'text': combined_texts,
        'type': labels,
        'dim1': coords[:, 0],
        'dim2': coords[:, 1],
        'dim3': coords[:, 2]
    })

    # Save results
    results.to_csv('combined_mds_coordinates.csv', index=False)
    print("\nSaved: combined_mds_coordinates.csv")

    return results

def visualize_combined_space(results=None):
    """
    Create visualizations comparing ingroup and outgroup positions.
    """
    if results is None:
        results = pd.read_csv('combined_mds_coordinates.csv')

    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # 1. Scatter plots colored by type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dim_pairs = [('dim1', 'dim2'), ('dim1', 'dim3'), ('dim2', 'dim3')]
    colors = {'ingroup': 'blue', 'outgroup': 'red'}

    for i, (dim_x, dim_y) in enumerate(dim_pairs):
        for t in ['ingroup', 'outgroup']:
            subset = results[results['type'] == t]
            axes[i].scatter(subset[dim_x], subset[dim_y],
                           c=colors[t], alpha=0.25, s=8, label=t, edgecolors='none')

        axes[i].set_xlabel(f'Dimension {dim_x[-1]}')
        axes[i].set_ylabel(f'Dimension {dim_y[-1]}')
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[i].legend()

    plt.suptitle('Combined Ingroup-Outgroup MDS Space', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_scatter.png', dpi=150)
    plt.close()
    print("Saved: combined_scatter.png")

    # 2. Density comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for i, dim in enumerate(['dim1', 'dim2', 'dim3']):
        for t, color in colors.items():
            subset = results[results['type'] == t][dim]
            sns.kdeplot(data=subset, ax=axes[i], color=color, fill=True,
                       alpha=0.4, label=t, linewidth=2)

        axes[i].set_title(f'Dimension {i+1} Distribution: Ingroup vs Outgroup')
        axes[i].set_xlabel('MDS Score')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_density.png', dpi=150)
    plt.close()
    print("Saved: combined_density.png")

    # 3. Overlap analysis
    print("\n" + "="*50)
    print("OVERLAP ANALYSIS")
    print("="*50)

    for dim in ['dim1', 'dim2', 'dim3']:
        ingroup_vals = results[results['type'] == 'ingroup'][dim]
        outgroup_vals = results[results['type'] == 'outgroup'][dim]

        # Calculate overlap coefficient
        in_mean, in_std = ingroup_vals.mean(), ingroup_vals.std()
        out_mean, out_std = outgroup_vals.mean(), outgroup_vals.std()

        # Cohen's d (effect size for separation)
        pooled_std = np.sqrt((in_std**2 + out_std**2) / 2)
        cohens_d = (in_mean - out_mean) / pooled_std

        print(f"\n{dim}:")
        print(f"  Ingroup:  mean={in_mean:.3f}, std={in_std:.3f}")
        print(f"  Outgroup: mean={out_mean:.3f}, std={out_std:.3f}")
        print(f"  Cohen's d (separation): {cohens_d:.3f}")

        if abs(cohens_d) < 0.2:
            print(f"  → Near-complete overlap (d < 0.2)")
        elif abs(cohens_d) < 0.5:
            print(f"  → Substantial overlap (0.2 < d < 0.5)")
        elif abs(cohens_d) < 0.8:
            print(f"  → Moderate separation (0.5 < d < 0.8)")
        else:
            print(f"  → Strong separation (d > 0.8)")

    # 4. Extreme responses comparison
    print("\n" + "="*50)
    print("EXTREME RESPONSES AT DIMENSION POLES")
    print("="*50)

    for dim in ['dim1', 'dim2', 'dim3']:
        print(f"\n--- {dim} ---")

        # LOW pole
        low_extremes = results.nsmallest(10, dim)
        low_ingroup = len(low_extremes[low_extremes['type'] == 'ingroup'])
        low_outgroup = len(low_extremes[low_extremes['type'] == 'outgroup'])
        print(f"LOW pole (bottom 10): {low_ingroup} ingroup, {low_outgroup} outgroup")

        # HIGH pole
        high_extremes = results.nlargest(10, dim)
        high_ingroup = len(high_extremes[high_extremes['type'] == 'ingroup'])
        high_outgroup = len(high_extremes[high_extremes['type'] == 'outgroup'])
        print(f"HIGH pole (top 10): {high_ingroup} ingroup, {high_outgroup} outgroup")

        # Example responses at extremes
        print("\n  Example LOW pole responses:")
        for _, row in low_extremes.head(3).iterrows():
            text = row['text'][:80] + '...' if len(row['text']) > 80 else row['text']
            print(f"    [{row['type']}] {text}")

        print("\n  Example HIGH pole responses:")
        for _, row in high_extremes.head(3).iterrows():
            text = row['text'][:80] + '...' if len(row['text']) > 80 else row['text']
            print(f"    [{row['type']}] {text}")

def test_mirroring_hypothesis(results=None):
    """
    Test whether ingroup and outgroup descriptions mirror each other at the poles.
    """
    if results is None:
        results = pd.read_csv('combined_mds_coordinates.csv')

    print("\n" + "="*60)
    print("MIRRORING HYPOTHESIS TEST")
    print("="*60)
    print("\nHypothesis: Ingroups and outgroups occupy opposite poles")
    print("(e.g., 'educated critical thinkers' vs 'uneducated sheep')")

    # For each dimension, check if types cluster at opposite ends
    for dim in ['dim1', 'dim2', 'dim3']:
        ingroup = results[results['type'] == 'ingroup'][dim]
        outgroup = results[results['type'] == 'outgroup'][dim]

        # Test: do they have significantly different means?
        from scipy.stats import ttest_ind, mannwhitneyu

        t_stat, t_pval = ttest_ind(ingroup, outgroup)
        u_stat, u_pval = mannwhitneyu(ingroup, outgroup, alternative='two-sided')

        print(f"\n{dim}:")
        print(f"  Ingroup mean:  {ingroup.mean():.4f}")
        print(f"  Outgroup mean: {outgroup.mean():.4f}")
        print(f"  Difference:    {ingroup.mean() - outgroup.mean():.4f}")
        print(f"  t-test p-value: {t_pval:.4f}")
        print(f"  Mann-Whitney p: {u_pval:.4f}")

        if t_pval < 0.05:
            if ingroup.mean() > outgroup.mean():
                print(f"  → Significant: Ingroups higher on {dim}")
            else:
                print(f"  → Significant: Outgroups higher on {dim}")
        else:
            print(f"  → Not significant: Types overlap on {dim}")

def main():
    """Run complete combined analysis."""
    print("="*60)
    print("COMBINED INGROUP-OUTGROUP MDS ANALYSIS")
    print("="*60)

    # Check if we need to run MDS or just visualize
    if not Path('combined_mds_coordinates.csv').exists():
        print("\nRunning new MDS analysis...")
        results = run_combined_mds()
    else:
        print("\nLoading existing combined_mds_coordinates.csv...")
        results = pd.read_csv('combined_mds_coordinates.csv')
        print(f"Loaded {len(results)} responses")

    print("\nGenerating visualizations...")
    visualize_combined_space(results)

    print("\nRunning statistical tests...")
    test_mirroring_hypothesis(results)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
