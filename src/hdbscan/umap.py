"""
NBA Player Clustering — UMAP + HDBSCAN Parameter Sweep
=======================================================
Tries all combinations of UMAP and HDBSCAN parameters,
scores each with silhouette, and saves the best result.

Run: python phase2_umap_sweep.py

Install:
    pip install umap-learn hdbscan matplotlib seaborn pandas scikit-learn
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan.umap as umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACE — edit these lists to control what gets tested
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_SETS = {
    "base": [
        'USG_PCT', 'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'TS_PCT', 'REB_PCT',
    ],
    "with_pie": [
        'USG_PCT', 'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'TS_PCT', 'REB_PCT', 'PIE',
    ],
    "shooting_focus": [
        'USG_PCT', 'AST_PCT', 'TS_PCT', 'EFG_PCT', 'REB_PCT', 'OREB_PCT',
    ],
}

UMAP_PARAMS = [
    {"n_neighbors": 10, "min_dist": 0.05},
    {"n_neighbors": 10, "min_dist": 0.10},
    {"n_neighbors": 15, "min_dist": 0.05},
    {"n_neighbors": 15, "min_dist": 0.10},
    {"n_neighbors": 20, "min_dist": 0.10},
    {"n_neighbors": 20, "min_dist": 0.20},
]

HDBSCAN_PARAMS = [
    {"min_cluster_size": 20, "min_samples": 3},
    {"min_cluster_size": 20, "min_samples": 5},
    {"min_cluster_size": 25, "min_samples": 5},
    {"min_cluster_size": 30, "min_samples": 5},
    {"min_cluster_size": 30, "min_samples": 8},
    {"min_cluster_size": 35, "min_samples": 5},
]

MIN_GP             = 15
MIN_VALID_CLUSTERS = 3    # skip results with fewer than this many clusters
MAX_VALID_CLUSTERS = 12   # skip results with more than this many clusters
MAX_NOISE_RATIO    = 0.30 # skip results where >30% of players are noise


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_players(min_gp=MIN_GP):
    players = pd.read_csv('player_stats.csv')
    players['PLAYER_ID'] = players['PLAYER_ID'].astype(str)
    df = players[players['GP'] >= min_gp].copy().reset_index(drop=True)
    print(f"Loaded {len(df)} players with {min_gp}+ games\n")
    return df


def scale_features(df, features):
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def run_umap(X_scaled, n_neighbors, min_dist, random_state=42):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    return reducer.fit_transform(X_scaled)


def run_hdbscan(embedding, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    return clusterer.fit_predict(embedding)


def score_result(embedding, labels, n_players):
    """
    Returns a dict of metrics for a given clustering result.
    silhouette is computed on the UMAP embedding (2D), excluding noise.
    """
    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    noise_ratio = n_noise / n_players

    # Not enough clusters or too much noise → invalid
    if n_clusters < MIN_VALID_CLUSTERS or n_clusters > MAX_VALID_CLUSTERS:
        return None
    if noise_ratio > MAX_NOISE_RATIO:
        return None

    # Silhouette only on non-noise points
    mask = labels != -1
    if mask.sum() < 10:
        return None

    sil = silhouette_score(embedding[mask], labels[mask])

    # Penalise tiny clusters (< 15 players) — they're usually noise
    cluster_sizes  = [np.sum(labels == c) for c in set(labels) if c != -1]
    smallest       = min(cluster_sizes)
    size_penalty   = 0.05 if smallest < 15 else 0.0

    return {
        "n_clusters":   n_clusters,
        "n_noise":      n_noise,
        "noise_ratio":  round(noise_ratio, 3),
        "silhouette":   round(sil, 4),
        "score":        round(sil - size_penalty, 4),   # adjusted score
        "smallest_cluster": smallest,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(df):
    results = []
    total   = len(FEATURE_SETS) * len(UMAP_PARAMS) * len(HDBSCAN_PARAMS)
    n       = 0

    for feat_name, features in FEATURE_SETS.items():
        # Check all needed columns exist
        available = [f for f in features if f in df.columns]
        if len(available) < len(features):
            missing = set(features) - set(available)
            print(f"  Skipping feature set '{feat_name}' — missing columns: {missing}")
            continue

        X_scaled = scale_features(df, available)

        for u_params in UMAP_PARAMS:
            # Run UMAP once per (feature_set, umap_params) combo — reuse for all HDBSCAN params
            embedding = run_umap(X_scaled, **u_params)

            for h_params in HDBSCAN_PARAMS:
                n += 1
                labels  = run_hdbscan(embedding, **h_params)
                metrics = score_result(embedding, labels, len(df))

                tag = (f"feat={feat_name} | "
                       f"nn={u_params['n_neighbors']} md={u_params['min_dist']} | "
                       f"mcs={h_params['min_cluster_size']} ms={h_params['min_samples']}")

                if metrics is None:
                    status = "SKIP"
                    print(f"  [{n:>3}/{total}] {tag}  →  skipped")
                else:
                    status = "OK"
                    print(f"  [{n:>3}/{total}] {tag}  →  "
                          f"k={metrics['n_clusters']}  "
                          f"noise={metrics['n_noise']}  "
                          f"sil={metrics['silhouette']:.4f}  "
                          f"score={metrics['score']:.4f}")

                results.append({
                    "status":       status,
                    "features":     feat_name,
                    "n_neighbors":  u_params['n_neighbors'],
                    "min_dist":     u_params['min_dist'],
                    "min_cluster_size": h_params['min_cluster_size'],
                    "min_samples":  h_params['min_samples'],
                    **(metrics or {})
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT BEST RESULT
# ══════════════════════════════════════════════════════════════════════════════

def plot_best(df, best_row):
    features   = FEATURE_SETS[best_row['features']]
    available  = [f for f in features if f in df.columns]
    X_scaled   = scale_features(df, available)
    embedding  = run_umap(X_scaled,
                          n_neighbors=int(best_row['n_neighbors']),
                          min_dist=float(best_row['min_dist']))
    labels     = run_hdbscan(embedding,
                             min_cluster_size=int(best_row['min_cluster_size']),
                             min_samples=int(best_row['min_samples']))

    df = df.copy()
    df['cluster'] = labels
    df['umap_x']  = embedding[:, 0]
    df['umap_y']  = embedding[:, 1]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    palette    = sns.color_palette("tab10", n_clusters)

    fig, ax = plt.subplots(figsize=(13, 9))

    noise = df[df['cluster'] == -1]
    ax.scatter(noise['umap_x'], noise['umap_y'],
               color='lightgrey', s=30, alpha=0.4, label='Noise', zorder=1)

    for i, c in enumerate(sorted([x for x in df['cluster'].unique() if x != -1])):
        sub = df[df['cluster'] == c]
        ax.scatter(sub['umap_x'], sub['umap_y'],
                   color=palette[i], s=55, alpha=0.75,
                   edgecolors='white', linewidths=0.4,
                   label=f'Cluster {c} (n={len(sub)})', zorder=2)
        for _, row in sub.sort_values('PIE', ascending=False).head(3).iterrows():
            ax.annotate(row['PLAYER_NAME'].split()[-1],
                        (row['umap_x'], row['umap_y']),
                        fontsize=7.5, alpha=0.9,
                        xytext=(4, 4), textcoords='offset points')

    ax.set_title(
        f"Best Configuration — UMAP + HDBSCAN\n"
        f"features={best_row['features']}  nn={best_row['n_neighbors']}  "
        f"md={best_row['min_dist']}  mcs={best_row['min_cluster_size']}  "
        f"ms={best_row['min_samples']}\n"
        f"k={n_clusters}  noise={int(best_row['n_noise'])}  "
        f"silhouette={best_row['silhouette']:.4f}",
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('umap_best_result.png', dpi=150)
    plt.close()
    print("  Saved umap_best_result.png")

    # Print cluster summary
    print(f"\n── Cluster Summary (best config) {'─'*40}")
    print(f"{'Cluster':>8}  {'Size':>5}  {'USG':>6}  {'AST':>6}  "
          f"{'OREB':>6}  {'DREB':>6}  {'TS':>6}  Top players")
    print("-" * 95)
    for c in sorted(df['cluster'].unique()):
        sub   = df[df['cluster'] == c]
        label = "NOISE" if c == -1 else str(c)
        top   = sub.sort_values('PIE', ascending=False).head(4)['PLAYER_NAME'].tolist()
        cols  = [f for f in ['USG_PCT','AST_PCT','OREB_PCT','DREB_PCT','TS_PCT'] if f in sub.columns]
        means = sub[cols].mean()
        print(f"  {label:>6}  {len(sub):>5}  "
              + "  ".join(f"{means.get(c2, 0):>6.3f}" for c2 in
                          ['USG_PCT','AST_PCT','OREB_PCT','DREB_PCT','TS_PCT'])
              + f"  {', '.join(top)}")

    # Save final assignments
    df.to_csv('player_archetypes_best.csv', index=False)
    print("  Saved player_archetypes_best.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_players()

    print(f"Running sweep — {len(FEATURE_SETS)} feature sets × "
          f"{len(UMAP_PARAMS)} UMAP configs × "
          f"{len(HDBSCAN_PARAMS)} HDBSCAN configs = "
          f"{len(FEATURE_SETS)*len(UMAP_PARAMS)*len(HDBSCAN_PARAMS)} total\n")

    results_df = run_sweep(df)

    # Save full results
    results_df.to_csv('sweep_results.csv', index=False)
    print("\n  Saved sweep_results.csv")

    # Filter valid and rank
    valid = results_df[results_df['status'] == 'OK'].sort_values('score', ascending=False)

    print(f"\n{'='*72}")
    print("TOP 10 CONFIGURATIONS BY SCORE")
    print(f"{'='*72}")
    cols = ['features', 'n_neighbors', 'min_dist', 'min_cluster_size',
            'min_samples', 'n_clusters', 'n_noise', 'silhouette', 'score']
    print(valid[cols].head(10).to_string(index=False))

    if valid.empty:
        print("\nNo valid configurations found — try relaxing MIN/MAX_VALID_CLUSTERS or MAX_NOISE_RATIO")
        return results_df, None

    best = valid.iloc[0]
    print(f"\n{'='*72}")
    print(f"BEST CONFIG:")
    print(f"  features        = {best['features']}")
    print(f"  n_neighbors     = {int(best['n_neighbors'])}")
    print(f"  min_dist        = {best['min_dist']}")
    print(f"  min_cluster_size= {int(best['min_cluster_size'])}")
    print(f"  min_samples     = {int(best['min_samples'])}")
    print(f"  → {int(best['n_clusters'])} clusters  |  "
          f"{int(best['n_noise'])} noise  |  "
          f"silhouette={best['silhouette']:.4f}")
    print(f"{'='*72}")

    print("\nPlotting best result...")
    final_df = plot_best(df, best)

    return results_df, final_df
