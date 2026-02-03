# Methodological Response: MDS Approach for Political Identity Analysis

## Question 1: Why Sentence Transformers Instead of Quanteda/cmdscale?

### The Traditional Approach (quanteda + cmdscale)

The traditional text-as-data pipeline in quanteda follows this logic:
1. Tokenize texts → Document-Feature Matrix (DFM)
2. Compute distances between documents (e.g., cosine, euclidean on TF-IDF)
3. Apply cmdscale() for MDS

**This works well when:**
- Documents are long enough to have meaningful word overlap
- The vocabulary is rich and varied
- Word co-occurrence patterns capture semantic meaning

### Why It Fails for Our Data

Our responses have a **median length of ~6 words**. Consider these two semantically similar responses:

| Response A | Response B |
|------------|------------|
| "Vzdělaní lidé, kteří si ověřují informace" | "Inteligentní, kriticky myslící, fakta si kontrolují" |
| (Educated people who verify information) | (Intelligent, critically thinking, check facts) |

**Under quanteda/DFM:**
- Response A tokens: {vzdělaní, lidé, kteří, ověřují, informace}
- Response B tokens: {inteligentní, kriticky, myslící, fakta, kontrolují}
- **Word overlap: ZERO** → Large distance despite identical meaning

**Under sentence transformer:**
- Both responses are encoded into 384-dimensional semantic vectors
- The model "understands" that "ověřují informace" ≈ "fakta kontrolují"
- **Cosine similarity: HIGH** → Small distance, correctly capturing semantic similarity

### Technical Comparison

| Aspect | quanteda/DFM | Sentence Transformer |
|--------|--------------|---------------------|
| Similarity basis | Word overlap | Semantic meaning |
| Short text handling | Poor (sparse vectors) | Excellent (dense vectors) |
| Synonyms | Treated as different | Recognized as similar |
| Cross-linguistic | Requires translation | Native multilingual |
| Czech morphology | Requires heavy lemmatization | Handles inflection implicitly |

### Empirical Support

Reimers & Gurevych (2019) showed that sentence transformers outperform bag-of-words approaches for semantic textual similarity, especially for short texts. For texts under 20 words, the improvement is substantial.

### Our Specific Advantages

1. **Czech morphological complexity**: Czech has 7 grammatical cases. "Vzdělaní lidé" and "vzdělaným lidem" are the same concept but different tokens. Sentence transformers handle this implicitly.

2. **Lemmatization limitations**: Even with UDPipe lemmatization, quanteda would miss semantic equivalences like:
   - "kritické myšlení" ≈ "analytický přístup"
   - "ověřují fakta" ≈ "kontrolují informace"

3. **Dense representation**: A 6-word response produces a 384-dimensional dense vector (sentence transformer) vs. a ~5,000-dimensional sparse vector with only 6 non-zero entries (DFM).

### Conclusion for Question 1

For short-text semantic similarity, sentence transformers are methodologically superior. The quanteda/cmdscale approach would produce a distance matrix dominated by vocabulary differences rather than meaning differences. Our goal is to find respondents who *describe their groups similarly*, not respondents who *use the same words*.

---

## Question 2: Visualization of Response Distributions on Dimensions

### Recommended Visualizations

#### 2.1 Density Plots by Dimension

For each MDS dimension, show the distribution of all responses:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load MDS coordinates
df = pd.read_csv('mds_coordinates.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, dim in enumerate(['mds_dim1', 'mds_dim2', 'mds_dim3']):
    sns.kdeplot(data=df, x=dim, ax=axes[i], fill=True, alpha=0.5)
    axes[i].set_title(f'Dimension {i+1} Distribution')
    axes[i].set_xlabel(f'MDS Dimension {i+1}')

plt.tight_layout()
plt.savefig('dimension_distributions.png', dpi=150)
```

#### 2.2 Scatter Plots with Labeled Poles

Show 2D projections with extreme responses labeled:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dim1 vs Dim2
axes[0].scatter(df['mds_dim1'], df['mds_dim2'], alpha=0.3, s=10)
axes[0].set_xlabel('Dim 1: Analytical ← → Ordinary')
axes[0].set_ylabel('Dim 2: Foreign Policy ← → Tribal')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Dim1 vs Dim3
axes[1].scatter(df['mds_dim1'], df['mds_dim3'], alpha=0.3, s=10)
axes[1].set_xlabel('Dim 1: Analytical ← → Ordinary')
axes[1].set_ylabel('Dim 3: Belief-based ← → Social Position')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Dim2 vs Dim3
axes[2].scatter(df['mds_dim2'], df['mds_dim3'], alpha=0.3, s=10)
axes[2].set_xlabel('Dim 2: Foreign Policy ← → Tribal')
axes[2].set_ylabel('Dim 3: Belief-based ← → Social Position')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('dimension_scatterplots.png', dpi=150)
```

#### 2.3 Party-Colored Distribution

Show how different party supporters distribute across dimensions:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

parties = ['ANO', 'ODS', 'Pirati', 'SPD', 'STAN']
colors = {'ANO': 'purple', 'ODS': 'blue', 'Pirati': 'black',
          'SPD': 'red', 'STAN': 'green'}

for i, dim in enumerate(['mds_dim1', 'mds_dim2', 'mds_dim3']):
    for party in parties:
        party_data = df[df['party'] == party][dim]
        sns.kdeplot(data=party_data, ax=axes[i], label=party,
                    color=colors[party], alpha=0.7)
    axes[i].set_title(f'Dimension {i+1} by Party')
    axes[i].legend()

plt.tight_layout()
plt.savefig('dimension_by_party.png', dpi=150)
```

#### 2.4 Annotated Extreme Responses

Create a visualization showing actual response text at the poles:

```python
fig, ax = plt.subplots(figsize=(12, 10))

ax.scatter(df['mds_dim1'], df['mds_dim2'], alpha=0.2, s=5, c='gray')

# Find and annotate extremes
extremes = {
    'low_dim1': df.nsmallest(3, 'mds_dim1'),
    'high_dim1': df.nlargest(3, 'mds_dim1'),
    'low_dim2': df.nsmallest(3, 'mds_dim2'),
    'high_dim2': df.nlargest(3, 'mds_dim2'),
}

for key, subset in extremes.items():
    for _, row in subset.iterrows():
        text = row['ingroup_text'][:50] + '...' if len(row['ingroup_text']) > 50 else row['ingroup_text']
        ax.annotate(text, (row['mds_dim1'], row['mds_dim2']),
                    fontsize=7, alpha=0.8)

ax.set_xlabel('Dimension 1: Analytical ← → Ordinary Middle Class')
ax.set_ylabel('Dimension 2: Foreign Policy ← → Tribal/Affective')
plt.savefig('annotated_scatterplot.png', dpi=150, bbox_inches='tight')
```

---

## Question 3: Analyzing Ingroups and Outgroups Together

### Rationale

Your colleague's intuition is methodologically sound. If we analyze ingroups and outgroups in a **single MDS space**, we can test whether:

1. **Mirror hypothesis**: Do ingroup and outgroup descriptions occupy opposite poles?
2. **Asymmetry**: Are outgroups described differently than ingroups (e.g., more extreme language)?
3. **Shared structure**: Do the same dimensions emerge for both?

### Implementation

```python
from sentence_transformers import SentenceTransformer
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('clean_data_with_improved_lemmas.csv')

# Prepare both ingroup and outgroup texts
ingroup_texts = df['ingroup_lemma'].dropna().tolist()
outgroup_texts = df['outgroup_lemma'].dropna().tolist()

# Create combined dataset with labels
combined_texts = ingroup_texts + outgroup_texts
labels = ['ingroup'] * len(ingroup_texts) + ['outgroup'] * len(outgroup_texts)

# Encode all texts
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(combined_texts, show_progress_bar=True)

# Compute distance matrix
dist_matrix = pairwise_distances(embeddings, metric='cosine')

# Fit MDS
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, n_jobs=-1)
coords = mds.fit_transform(dist_matrix)

# Create results dataframe
results = pd.DataFrame({
    'text': combined_texts,
    'type': labels,
    'dim1': coords[:, 0],
    'dim2': coords[:, 1],
    'dim3': coords[:, 2]
})

results.to_csv('combined_mds_coordinates.csv', index=False)
```

### Visualization of Combined Space

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Color by ingroup/outgroup
colors = {'ingroup': 'blue', 'outgroup': 'red'}

for i, (dim_x, dim_y) in enumerate([('dim1', 'dim2'), ('dim1', 'dim3'), ('dim2', 'dim3')]):
    for t in ['ingroup', 'outgroup']:
        subset = results[results['type'] == t]
        axes[i].scatter(subset[dim_x], subset[dim_y],
                       c=colors[t], alpha=0.3, s=10, label=t)
    axes[i].set_xlabel(dim_x)
    axes[i].set_ylabel(dim_y)
    axes[i].legend()

plt.suptitle('Combined Ingroup-Outgroup MDS Space')
plt.tight_layout()
plt.savefig('combined_mds_space.png', dpi=150)
```

### What to Look For

1. **Separation**: If ingroups and outgroups form distinct clusters, people use fundamentally different language for each.

2. **Overlap at poles**: If they meet at the extremes (as your colleague hypothesized), it would suggest that extreme ingroup descriptions mirror extreme outgroup descriptions (e.g., "educated critical thinkers" vs "uneducated sheep").

3. **Dimension interpretation**: In a combined space, dimensions might capture:
   - **Valence**: Positive (ingroup) ↔ Negative (outgroup)
   - **Specificity**: Concrete descriptions ↔ Abstract/tribal
   - **Threat type**: Cultural ↔ Economic

### Methodological Considerations

**Advantages of combined analysis:**
- Direct comparison of ingroup/outgroup framing
- Single coherent semantic space
- Can test specific hypotheses about mirroring

**Advantages of separate analysis (current approach):**
- Each space optimally represents its own variation
- Avoids dimension "contamination" across types
- Clearer interpretation of dimensions within each type

**Recommendation:** Run both analyses. Use separate MDS for primary results (cleaner interpretation) and combined MDS as supplementary analysis to test the mirroring hypothesis.

---

## Summary of Responses

| Question | Short Answer |
|----------|-------------|
| Why sentence transformer? | Short texts lack word overlap; we need semantic similarity, not vocabulary similarity |
| Visualizations? | Density plots, 2D scatter plots with pole labels, party-colored distributions |
| Combined analysis? | Yes, as supplementary—tests whether ingroups/outgroups occupy opposite poles |

---

## Code for Complete Visualization Suite

See the attached Python script `visualization_suite.py` for a complete implementation of all visualizations discussed above.

*Prepared in response to methodological questions from collaborators*
