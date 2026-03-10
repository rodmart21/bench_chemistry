"""
NBA Lineup Chemistry — Cluster Evaluation
==========================================
Tries K from 2 to 12, computes multiple validation metrics,
and visualises each solution so you can pick the best K.

Metrics computed per K:
    - Inertia (elbow)
    - Silhouette Score        (higher = better, max 1.0)
    - Davies-Bouldin Index    (lower = better, min 0)
    - Calinski-Harabasz Index (higher = better)
    - Per-cluster silhouette  (reveals which clusters are weak)

Outputs:
    cluster_evaluation_metrics.png   — 4-panel metric curves
    cluster_silhouette_k{K}.png      — per-cluster silhouette bar for each K
    cluster_pca_grid.png             — PCA plots for every K in one grid
    cluster_evaluation_summary.csv   — numeric table of all metrics
    player_archetypes_k{K}.csv       — player assignments for any chosen K

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────

K_RANGE     = range(2, 13)   # try K = 2 through 12
N_INIT      = 20             # KMeans restarts — more = more stable
RANDOM_STATE = 42
MIN_GP      = 15

CLUSTER_FEATURES = [
    'USG_PCT', 'AST_PCT', 'OREB_PCT', 'DREB_PCT',
    'TS_PCT',  'PIE',     'REB_PCT',
]


# ── LOAD & PREP ───────────────────────────────────────────────────────────────

def load_and_scale():
    print("Loading player_stats.csv...")
    players = pd.read_csv('player_stats.csv')
    players['PLAYER_ID'] = players['PLAYER_ID'].astype(str)

    df = players[players['GP'] >= MIN_GP].copy()
    print(f"  {len(df)} players with {MIN_GP}+ games")

    X = df[CLUSTER_FEATURES].copy()
    for col in CLUSTER_FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


# ── METRIC SWEEP ──────────────────────────────────────────────────────────────

def sweep_k(X_scaled):
    """
    Runs KMeans for every K in K_RANGE and collects all metrics.
    Returns a DataFrame with one row per K.
    """
    results = []
    all_labels = {}

    for k in K_RANGE:
        print(f"  Fitting K={k}...", end=' ')
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X_scaled)

        inertia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels)
        db        = davies_bouldin_score(X_scaled, labels)
        ch        = calinski_harabasz_score(X_scaled, labels)
        min_sil   = silhouette_samples(X_scaled, labels)

        # Weakest cluster = cluster with lowest avg silhouette
        cluster_sils = [min_sil[labels == c].mean() for c in range(k)]
        weakest_sil  = min(cluster_sils)
        smallest_cluster = min(np.bincount(labels))

        print(f"sil={sil:.3f}  DB={db:.3f}  CH={ch:.0f}  weakest_cluster_sil={weakest_sil:.3f}")

        results.append({
            'k':               k,
            'inertia':         inertia,
            'silhouette':      sil,
            'davies_bouldin':  db,
            'calinski_harabasz': ch,
            'weakest_cluster_sil': weakest_sil,
            'smallest_cluster_size': smallest_cluster,
        })
        all_labels[k] = labels

    return pd.DataFrame(results), all_labels


# ── PLOT 1: Metric curves ─────────────────────────────────────────────────────

def plot_metric_curves(summary_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Cluster Quality Metrics vs K", fontsize=15, fontweight='bold', y=1.01)
    ks = summary_df['k']

    # Inertia
    axes[0,0].plot(ks, summary_df['inertia'], 'o-', color='steelblue', linewidth=2)
    axes[0,0].set_title('Inertia (Elbow)'); axes[0,0].set_xlabel('K')
    axes[0,0].set_ylabel('Inertia'); axes[0,0].grid(alpha=0.3)

    # Silhouette (main + weakest cluster)
    axes[0,1].plot(ks, summary_df['silhouette'], 'o-', color='green', linewidth=2, label='Overall')
    axes[0,1].plot(ks, summary_df['weakest_cluster_sil'], 's--', color='tomato', linewidth=2, label='Weakest cluster')
    axes[0,1].axhline(0, color='black', linewidth=0.8, linestyle=':')
    best_k = summary_df.loc[summary_df['silhouette'].idxmax(), 'k']
    axes[0,1].axvline(best_k, color='green', linewidth=1.2, linestyle='--', alpha=0.6, label=f'Best K={best_k}')
    axes[0,1].set_title('Silhouette Score (↑ better)'); axes[0,1].set_xlabel('K')
    axes[0,1].set_ylabel('Silhouette'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # Davies-Bouldin
    axes[1,0].plot(ks, summary_df['davies_bouldin'], 'o-', color='darkorange', linewidth=2)
    best_db_k = summary_df.loc[summary_df['davies_bouldin'].idxmin(), 'k']
    axes[1,0].axvline(best_db_k, color='darkorange', linewidth=1.2, linestyle='--', alpha=0.6, label=f'Best K={best_db_k}')
    axes[1,0].set_title('Davies-Bouldin Index (↓ better)'); axes[1,0].set_xlabel('K')
    axes[1,0].set_ylabel('DB Index'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    # Calinski-Harabasz
    axes[1,1].plot(ks, summary_df['calinski_harabasz'], 'o-', color='purple', linewidth=2)
    best_ch_k = summary_df.loc[summary_df['calinski_harabasz'].idxmax(), 'k']
    axes[1,1].axvline(best_ch_k, color='purple', linewidth=1.2, linestyle='--', alpha=0.6, label=f'Best K={best_ch_k}')
    axes[1,1].set_title('Calinski-Harabasz Index (↑ better)'); axes[1,1].set_xlabel('K')
    axes[1,1].set_ylabel('CH Index'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved cluster_evaluation_metrics.png")


# ── PLOT 2: Per-cluster silhouette bars (one plot per K) ─────────────────────

def plot_silhouette_per_k(X_scaled, all_labels, ks_to_plot=None):
    """
    For each K, shows a bar chart of avg silhouette per cluster.
    Bars below 0 = clusters that should not exist — their members
    would be better assigned elsewhere.
    """
    if ks_to_plot is None:
        ks_to_plot = list(K_RANGE)

    n   = len(ks_to_plot)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, k in enumerate(ks_to_plot):
        ax     = axes[idx]
        labels = all_labels[k]
        sample_sils = silhouette_samples(X_scaled, labels)
        palette = cm.tab10(np.linspace(0, 1, k))

        cluster_means = []
        for c in range(k):
            mask = labels == c
            avg  = sample_sils[mask].mean()
            n_c  = mask.sum()
            color = palette[c] if avg >= 0 else 'tomato'
            ax.barh(c, avg, color=color, edgecolor='white', height=0.7)
            ax.text(avg + 0.005, c, f'{avg:.3f}  (n={n_c})', va='center', fontsize=8)
            cluster_means.append(avg)

        overall = np.mean(cluster_means)
        ax.axvline(overall, color='black', linewidth=1.2, linestyle='--', label=f'Avg={overall:.3f}')
        ax.axvline(0,       color='red',   linewidth=0.8, linestyle=':')
        ax.set_title(f'K={k}  |  overall sil={overall:.3f}', fontweight='bold')
        ax.set_xlabel('Silhouette Score'); ax.set_ylabel('Cluster')
        ax.set_yticks(range(k)); ax.set_yticklabels([f'C{i}' for i in range(k)])
        ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-0.2, 0.6)

    for idx in range(len(ks_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Cluster Silhouette Scores by K\n(Red bars = cluster below 0 — unreliable)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_silhouette_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cluster_silhouette_bars.png")


# ── PLOT 3: PCA grid for all K values ────────────────────────────────────────

def plot_pca_grid(X_scaled, all_labels, players_df):
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_ * 100

    ks   = list(K_RANGE)
    cols = 3
    rows = (len(ks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, k in enumerate(ks):
        ax     = axes[idx]
        labels = all_labels[k]
        palette = sns.color_palette("tab10", k)
        sil    = silhouette_score(X_scaled, labels)

        for c in range(k):
            mask = labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=palette[c], alpha=0.6, s=25, edgecolors='white', linewidths=0.3,
                       label=f'C{c} (n={mask.sum()})')

        ax.set_title(f'K={k}  |  sil={sil:.3f}', fontweight='bold', fontsize=10)
        ax.set_xlabel(f'PC1 ({var[0]:.0f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({var[1]:.0f}%)', fontsize=8)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    for idx in range(len(ks), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("PCA Projection for Each K Value\n(Colour = cluster assignment)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_pca_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cluster_pca_grid.png")


# ── EXPORT player assignments for a chosen K ─────────────────────────────────

def export_player_archetypes(players_df, X_scaled, k, scaler):
    """
    Fits final KMeans for a given K and saves player assignments.
    Call this after you've decided on your preferred K.
    """
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_scaled)

    df = players_df.copy()
    df['archetype'] = labels

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=CLUSTER_FEATURES
    )
    centroids.index.name = 'cluster'

    print(f"\n── Centroids for K={k} ─────────────────────────────────────────────")
    print(centroids.round(3).to_string())
    print(f"\n── Top 5 players per cluster (K={k}) ──────────────────────────────")
    for c in range(k):
        top = df[df['archetype'] == c].sort_values('PIE', ascending=False).head(5)['PLAYER_NAME'].tolist()
        print(f"  C{c}: {', '.join(top)}")

    fname = f'player_archetypes_k{k}.csv'
    df.to_csv(fname, index=False)
    print(f"\n  Saved {fname}")
    return df


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def print_summary(summary_df):
    print("\n" + "="*72)
    print("CLUSTER EVALUATION SUMMARY")
    print("="*72)
    print(summary_df.to_string(index=False))

    best_sil = summary_df.loc[summary_df['silhouette'].idxmax()]
    best_db  = summary_df.loc[summary_df['davies_bouldin'].idxmin()]
    best_ch  = summary_df.loc[summary_df['calinski_harabasz'].idxmax()]

    print(f"\n  Best K by Silhouette:        K={int(best_sil['k'])}  ({best_sil['silhouette']:.4f})")
    print(f"  Best K by Davies-Bouldin:    K={int(best_db['k'])}  ({best_db['davies_bouldin']:.4f})")
    print(f"  Best K by Calinski-Harabasz: K={int(best_ch['k'])}  ({best_ch['calinski_harabasz']:.1f})")
    print()
    print("  Tip: if all three agree → easy choice.")
    print("  If they disagree → look at cluster_silhouette_bars.png and")
    print("  cluster_pca_grid.png to pick the K that makes basketball sense.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    players_df, X_scaled, scaler = load_and_scale()

    print(f"\nRunning KMeans for K = {list(K_RANGE)}...")
    summary_df, all_labels = sweep_k(X_scaled)

    summary_df.to_csv('cluster_evaluation_summary.csv', index=False)
    print("\n  Saved cluster_evaluation_summary.csv")

    print("\nGenerating plots...")
    plot_metric_curves(summary_df)
    plot_silhouette_per_k(X_scaled, all_labels)
    plot_pca_grid(X_scaled, all_labels, players_df)

    print_summary(summary_df)

    # ── Once you pick a K, call this to export final assignments ──────────────
    # Change the number below to whichever K you decide on after reviewing plots
    CHOSEN_K = int(summary_df.loc[summary_df['silhouette'].idxmax(), 'k'])
    print(f"\nAuto-exporting best K by silhouette = {CHOSEN_K}...")
    export_player_archetypes(players_df, X_scaled, CHOSEN_K, scaler)

    print("\nDone. Files saved:")
    print("  cluster_evaluation_metrics.png  — 4 metric curves")
    print("  cluster_silhouette_bars.png     — per-cluster silhouette for every K")
    print("  cluster_pca_grid.png            — PCA view for every K")
    print("  cluster_evaluation_summary.csv  — numeric table")
    print(f"  player_archetypes_k{CHOSEN_K}.csv       — player assignments for best K")

    return summary_df, all_labels


if __name__ == '__main__':
    summary_df, all_labels = main()