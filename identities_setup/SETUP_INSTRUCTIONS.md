# How to Set Up Your New Repository

## Step 1: Clone your new repo locally

```bash
git clone https://github.com/Hrbkova/identities.git
cd identities
```

## Step 2: Copy these files into it

Copy the entire contents of this `identities_setup` folder into your cloned repo.

Your structure should look like:
```
identities/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── analysis/
│   ├── 01_preprocessing/
│   ├── 02_methods/
│   │   └── run_all_methods.py
│   └── 03_comparison/
├── results/
│   ├── figures/
│   └── tables/
├── docs/
└── notebooks/
```

## Step 3: Add your data

Copy your CSV files to `data/raw/`:
- `Czech_Transformed.csv`
- `Hungarian_Transformed.csv` (when ready)
- `Lithuanian_Transformed.csv` (when ready)

## Step 4: Commit and push

```bash
git add .
git commit -m "Set up project structure"
git push origin main
```

## Step 5: Add collaborators

On GitHub:
1. Go to Settings → Collaborators
2. Add your coauthors by username or email

## Step 6: Share with coauthors

Send them the repo link and this message:

---

**For coauthors:**

I've set up a GitHub repo for our identities project:
https://github.com/Hrbkova/identities

**To run the analysis:**
1. Clone the repo: `git clone https://github.com/Hrbkova/identities.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run methods comparison: `python analysis/02_methods/run_all_methods.py --data data/raw/Czech_Transformed.csv --text_col ingroup_lemma`

**Or use Google Colab:**
[Link to notebook - I'll add this]

Results will appear in `results/` and `docs/methods_comparison.md`.

---

## Alternative: Use Google Colab

If you prefer Colab:
1. Upload `run_all_methods.py` to Colab
2. Upload your data when prompted
3. Share the Colab link with coauthors

This way everyone can run the same analysis.
