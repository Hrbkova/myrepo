"""
Visualization Suite for MDS Analysis
=====================================
This script generates all visualizations for the MDS analysis of Czech political identity descriptions.

Run in Google Colab or locally with: python visualization_suite.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(filepath='mds_coordinates.csv'):
    """Load MDS coordinates data."""
    df = pd.read_csv(filepath)
    return df

def plot_dimension_distributions(df, output_dir='.'):
    """
    Plot 1: Kernel density plots for each MDS dimension.
    Shows the overall distribution of responses across each dimension.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    dim_labels = {
        'mds_dim1': 'Dim 1: Analytical ↔ Ordinary Middle Class',
        'mds_dim2': 'Dim 2: Foreign Policy ↔ Tribal/Affective',
        'mds_dim3': 'Dim 3: Belief-based ↔ Social Position'
    }

    for i, (dim, label) in enumerate(dim_labels.items()):
        if dim in df.columns:
            sns.kdeplot(data=df, x=dim, ax=axes[i], fill=True, alpha=0.5, color='steelblue')
            axes[i].set_title(label, fontsize=10)
            axes[i].set_xlabel('MDS Score')
            axes[i].set_ylabel('Density')

            # Add mean and std lines
            mean_val = df[dim].mean()
            std_val = df[dim].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
            axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'dimension_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dimension_distributions.png")

def plot_2d_scatterplots(df, output_dir='.'):
    """
    Plot 2: 2D scatter plots showing all pairwise dimension combinations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dim_pairs = [
        ('mds_dim1', 'mds_dim2', 'Analytical ↔ Ordinary', 'Foreign Policy ↔ Tribal'),
        ('mds_dim1', 'mds_dim3', 'Analytical ↔ Ordinary', 'Belief-based ↔ Social'),
        ('mds_dim2', 'mds_dim3', 'Foreign Policy ↔ Tribal', 'Belief-based ↔ Social')
    ]

    for i, (dim_x, dim_y, label_x, label_y) in enumerate(dim_pairs):
        if dim_x in df.columns and dim_y in df.columns:
            axes[i].scatter(df[dim_x], df[dim_y], alpha=0.3, s=15, c='steelblue', edgecolors='none')
            axes[i].set_xlabel(label_x, fontsize=10)
            axes[i].set_ylabel(label_y, fontsize=10)
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            # Add quadrant labels
            xlim = axes[i].get_xlim()
            ylim = axes[i].get_ylim()
            axes[i].text(xlim[0]*0.8, ylim[1]*0.8, 'Q1', fontsize=8, alpha=0.5)
            axes[i].text(xlim[1]*0.8, ylim[1]*0.8, 'Q2', fontsize=8, alpha=0.5)
            axes[i].text(xlim[0]*0.8, ylim[0]*0.8, 'Q3', fontsize=8, alpha=0.5)
            axes[i].text(xlim[1]*0.8, ylim[0]*0.8, 'Q4', fontsize=8, alpha=0.5)

    plt.suptitle('MDS Dimension Scatter Plots (Ingroup Descriptions)', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'dimension_scatterplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dimension_scatterplots.png")

def plot_party_distributions(df, output_dir='.'):
    """
    Plot 3: Dimension distributions colored by party affiliation.
    """
    if 'party' not in df.columns:
        print("No party column found, skipping party distribution plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Define party colors (Czech political parties)
    party_colors = {
        'ANO': '#261160',      # ANO purple
        'ODS': '#004494',      # ODS blue
        'Pirati': '#000000',   # Pirates black
        'SPD': '#003366',      # SPD dark blue
        'STAN': '#00A650',     # STAN green
        'KDU-CSL': '#FFD700',  # KDU yellow
        'TOP09': '#6C1D7C',    # TOP09 purple
        'KSCM': '#CC0000',     # KSCM red
        'CSSD': '#EC5800',     # CSSD orange
        'Other': '#808080',    # Gray for other
        'Unknown': '#C0C0C0'   # Light gray for unknown
    }

    dim_labels = ['Dim 1: Analytical ↔ Ordinary',
                  'Dim 2: Foreign Policy ↔ Tribal',
                  'Dim 3: Belief-based ↔ Social']

    # Get parties with at least 30 respondents
    party_counts = df['party'].value_counts()
    major_parties = party_counts[party_counts >= 30].index.tolist()

    for i, dim in enumerate(['mds_dim1', 'mds_dim2', 'mds_dim3']):
        if dim not in df.columns:
            continue

        for party in major_parties:
            party_data = df[df['party'] == party][dim]
            color = party_colors.get(party, '#808080')
            try:
                sns.kdeplot(data=party_data, ax=axes[i], label=f'{party} (n={len(party_data)})',
                           color=color, alpha=0.7, linewidth=2)
            except:
                pass

        axes[i].set_title(dim_labels[i], fontsize=10)
        axes[i].set_xlabel('MDS Score')
        axes[i].legend(fontsize=8, loc='upper right')

    plt.suptitle('MDS Dimensions by Party Affiliation', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'dimension_by_party.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dimension_by_party.png")

def plot_extreme_responses(df, output_dir='.'):
    """
    Plot 4: Scatter plot with extreme responses annotated.
    """
    if 'ingroup_text' not in df.columns:
        print("No ingroup_text column found, skipping extreme responses plot")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all points
    ax.scatter(df['mds_dim1'], df['mds_dim2'], alpha=0.2, s=8, c='lightgray', edgecolors='none')

    # Find extreme responses
    n_extremes = 5

    extremes = {
        'LOW Dim1 (Analytical)': df.nsmallest(n_extremes, 'mds_dim1'),
        'HIGH Dim1 (Ordinary)': df.nlargest(n_extremes, 'mds_dim1'),
        'LOW Dim2 (Foreign Policy)': df.nsmallest(n_extremes, 'mds_dim2'),
        'HIGH Dim2 (Tribal)': df.nlargest(n_extremes, 'mds_dim2'),
    }

    colors = {
        'LOW Dim1 (Analytical)': 'blue',
        'HIGH Dim1 (Ordinary)': 'red',
        'LOW Dim2 (Foreign Policy)': 'green',
        'HIGH Dim2 (Tribal)': 'orange',
    }

    for label, subset in extremes.items():
        ax.scatter(subset['mds_dim1'], subset['mds_dim2'],
                  c=colors[label], s=50, alpha=0.8, label=label, edgecolors='black')

        for _, row in subset.iterrows():
            text = str(row['ingroup_text'])[:60]
            if len(str(row['ingroup_text'])) > 60:
                text += '...'
            ax.annotate(text,
                       (row['mds_dim1'], row['mds_dim2']),
                       fontsize=6, alpha=0.9,
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Dimension 1: Analytical ← → Ordinary Middle Class', fontsize=11)
    ax.set_ylabel('Dimension 2: Foreign Policy ← → Tribal/Affective', fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)

    plt.title('MDS Space with Extreme Responses Annotated', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'annotated_extremes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: annotated_extremes.png")

def plot_dimension_histograms(df, output_dir='.'):
    """
    Plot 5: Histograms showing response counts at different dimension values.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    dim_info = [
        ('mds_dim1', 'Dimension 1', 'Analytical/Educated', 'Ordinary Middle Class'),
        ('mds_dim2', 'Dimension 2', 'Foreign Policy/Supranational', 'Tribal/Affective'),
        ('mds_dim3', 'Dimension 3', 'Belief-based', 'Social Position')
    ]

    for i, (dim, title, low_label, high_label) in enumerate(dim_info):
        if dim not in df.columns:
            continue

        axes[i].hist(df[dim], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        axes[i].set_xlabel('MDS Score')
        axes[i].set_ylabel('Number of Responses')
        axes[i].set_title(f'{title}: {low_label} ← → {high_label}')

        # Add pole labels
        xlim = axes[i].get_xlim()
        ylim = axes[i].get_ylim()
        axes[i].text(xlim[0] + 0.05*(xlim[1]-xlim[0]), ylim[1]*0.9,
                    f'← {low_label}', fontsize=9, color='darkblue')
        axes[i].text(xlim[1] - 0.25*(xlim[1]-xlim[0]), ylim[1]*0.9,
                    f'{high_label} →', fontsize=9, color='darkred')

        # Add statistics
        mean_val = df[dim].mean()
        std_val = df[dim].std()
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')
        axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'dimension_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dimension_histograms.png")

def create_summary_statistics(df, output_dir='.'):
    """
    Create summary statistics table for each dimension.
    """
    dims = ['mds_dim1', 'mds_dim2', 'mds_dim3']
    dims = [d for d in dims if d in df.columns]

    stats = df[dims].describe()

    # Add skewness and kurtosis
    from scipy import stats as scipy_stats
    for dim in dims:
        stats.loc['skewness', dim] = scipy_stats.skew(df[dim].dropna())
        stats.loc['kurtosis', dim] = scipy_stats.kurtosis(df[dim].dropna())

    # Rename for clarity
    stats.columns = ['Dim1: Analytical/Ordinary', 'Dim2: ForeignPolicy/Tribal', 'Dim3: Belief/Social']

    stats.to_csv(Path(output_dir) / 'dimension_statistics.csv')
    print("Saved: dimension_statistics.csv")
    print("\nDimension Statistics:")
    print(stats.round(4))
    return stats

def main():
    """Run all visualizations."""
    print("=" * 60)
    print("MDS Visualization Suite")
    print("=" * 60)

    # Try to load data
    try:
        df = load_data('mds_coordinates.csv')
        print(f"Loaded {len(df)} responses from mds_coordinates.csv")
    except FileNotFoundError:
        print("ERROR: mds_coordinates.csv not found. Please run MDS analysis first.")
        return

    print(f"Columns available: {df.columns.tolist()}")
    print()

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Run all visualization functions
    print("Generating visualizations...")
    print("-" * 40)

    plot_dimension_distributions(df, output_dir)
    plot_2d_scatterplots(df, output_dir)
    plot_party_distributions(df, output_dir)
    plot_extreme_responses(df, output_dir)
    plot_dimension_histograms(df, output_dir)
    create_summary_statistics(df, output_dir)

    print("-" * 40)
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
