"""
NBA Lineup Chemistry — Phase 2: Player Archetype Clustering
============================================================
Clusters all NBA players into role archetypes based on their
playing style, then enriches every lineup with archetype composition
data so Phase 3 can learn which archetype combos produce chemistry.

Outputs:
    player_archetypes.csv       — each player + their cluster label + stats
    lineups_with_archetypes.csv — lineups_with_synergy.csv enriched with
                                  archetype fingerprint per lineup

Run after phase1_synergy_delta.py has produced:
    - lineups_with_synergy.csv
    - player_stats.csv

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter

warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────

N_CLUSTERS   = 7       # 7 modern NBA archetypes (see labels below)
RANDOM_STATE = 42
MIN_GP       = 15      # ignore players with very few games (noisy stats)

# Features that define HOW a player plays (style + role, not just volume)
CLUSTER_FEATURES = [
    'USG_PCT',    # how much of the offense runs through them
    'AST_PCT',    # playmaking responsibility
    'OREB_PCT',   # offensive rebounding — paint presence
    'DREB_PCT',   # defensive rebounding
    'TS_PCT',     # shooting efficiency
    'PIE',        # holistic impact
    'REB_PCT',    # total rebounding
]

# Human-readable archetype labels — assign after inspecting cluster centroids
# The numbers will shift each run; reassign based on the centroid printout
ARCHETYPE_LABELS = {
    0: "3-and-D Wing",
    1: "Rim-Runner",
    2: "Secondary Creator",
    3: "Star Engine",          # Jokic, Giannis, Wemby, Luka
    4: "Versatile Big",
    5: "Glue / Role Player",
    6: "Primary Ball-Handler", # SGA, Cade, Mitchell
}

# ── STEP 1: Load data ─────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    players  = pd.read_csv('results/player_stats.csv')
    lineups  = pd.read_csv('results/synergy/lineups_with_synergy.csv')
    players['PLAYER_ID'] = players['PLAYER_ID'].astype(str)
    print(f"  {len(players)} player-season rows")
    print(f"  {len(lineups)} lineup rows")
    return players, lineups


# ── STEP 2: Cluster players ───────────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(4, 12)):
    """
    Plots inertia (elbow) and silhouette scores to help pick N_CLUSTERS.
    Optional — run once to validate the choice of 7.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, 'o-', color='steelblue')
    ax1.set_title('Elbow — Inertia vs K'); ax1.set_xlabel('K')
    ax2.plot(list(k_range), silhouettes, 'o-', color='darkorange')
    ax2.set_title('Silhouette Score vs K'); ax2.set_xlabel('K')
    plt.tight_layout()
    plt.savefig('optimal_k.png', dpi=150)
    plt.close()
    print("  Saved optimal_k.png — inspect to confirm N_CLUSTERS=7 is appropriate")


def cluster_players(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits K-Means on player style features.
    Returns the DataFrame with 'archetype' (int) and 'archetype_label' (str) columns.
    """
    # Filter: need enough games for stable stats
    df = players_df[players_df['GP'] >= MIN_GP].copy()
    print(f"\n  {len(df)} players with {MIN_GP}+ games (used for clustering)")

    # Fill rare NaNs with column median
    X = df[CLUSTER_FEATURES].copy()
    for col in CLUSTER_FEATURES:
        X[col] = X[col].fillna(X[col].median())

    # Scale to zero mean, unit variance
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional: uncomment to run elbow/silhouette analysis
    # find_optimal_k(X_scaled)

    # Fit K-Means
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    df['archetype'] = km.fit_predict(X_scaled)
    df['archetype_label'] = df['archetype'].map(ARCHETYPE_LABELS)

    # Print centroids so you can re-label ARCHETYPE_LABELS if needed
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=CLUSTER_FEATURES
    )
    centroids.index.name = 'cluster'
    print("\n── Cluster Centroids (original scale) ──────────────────────────────")
    print(centroids.round(3).to_string())
    print()

    # Print top 5 players per cluster
    print("── Top players per cluster (by PIE) ───────────────────────────────")
    for cluster_id in sorted(df['archetype'].unique()):
        label  = ARCHETYPE_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        sample = (df[df['archetype'] == cluster_id]
                    .sort_values('PIE', ascending=False)
                    .head(5)['PLAYER_NAME'].tolist())
        print(f"  [{cluster_id}] {label:25s}: {', '.join(sample)}")

    return df, scaler, km


# ── STEP 3: Visualise clusters with PCA ──────────────────────────────────────

def plot_clusters(players_with_archetypes: pd.DataFrame):
    """
    Projects the 7-feature space down to 2D with PCA for visualisation.
    """
    X = players_with_archetypes[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    df_plot = players_with_archetypes.copy()
    df_plot['pca_1'] = coords[:, 0]
    df_plot['pca_2'] = coords[:, 1]

    palette = sns.color_palette("tab10", N_CLUSTERS)
    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(df_plot['archetype'].unique()):
        label  = ARCHETYPE_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        subset = df_plot[df_plot['archetype'] == cluster_id]
        ax.scatter(subset['pca_1'], subset['pca_2'],
                   label=f"[{cluster_id}] {label}",
                   color=palette[cluster_id], alpha=0.65, s=60, edgecolors='white', linewidths=0.4)

    # Annotate a few notable players
    notable = (df_plot.sort_values('PIE', ascending=False)
                       .groupby('archetype').head(2))
    for _, row in notable.iterrows():
        ax.annotate(row['PLAYER_NAME'].split()[-1],
                    (row['pca_1'], row['pca_2']),
                    fontsize=7, alpha=0.8,
                    xytext=(4, 4), textcoords='offset points')

    var_explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
    ax.set_title("NBA Player Archetypes — PCA Projection", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('player_archetypes_pca.png', dpi=150)
    plt.close()
    print("  Saved player_archetypes_pca.png")


# ── STEP 4: Enrich lineups with archetype fingerprint ────────────────────────

def build_archetype_lookup(players_with_archetypes: pd.DataFrame) -> dict:
    """
    Returns {player_id: archetype_int} for all clustered players.
    """
    return dict(zip(
        players_with_archetypes['PLAYER_ID'],
        players_with_archetypes['archetype']
    ))


def compute_archetype_fingerprint(player_ids: list[str], lookup: dict):
    """
    Returns a sorted tuple of archetype ints for this lineup.
    e.g. (0, 1, 1, 3, 5) — sorted so order doesn't matter.
    Returns None if any player is missing from the lookup.
    """
    archetypes = [lookup.get(pid) for pid in player_ids]
    if any(a is None for a in archetypes):
        return None
    return tuple(sorted(archetypes))


def enrich_lineups(lineups_df: pd.DataFrame, archetype_lookup: dict) -> pd.DataFrame:
    """
    Adds archetype-level features to every lineup row.
    """
    df = lineups_df.copy()
    df['player_ids_list'] = df['player_ids_str'].apply(
        lambda s: str(s).split(',') if pd.notna(s) else []
    )

    # Archetype fingerprint — the core feature for Phase 3
    df['archetype_fingerprint'] = df['player_ids_list'].apply(
        lambda ids: compute_archetype_fingerprint(ids, archetype_lookup)
    )
    df['archetype_fingerprint_str'] = df['archetype_fingerprint'].apply(
        lambda x: str(x) if x is not None else None
    )

    # Archetype-level lineup features
    def archetype_features(player_ids):
        archetypes = [archetype_lookup.get(pid) for pid in player_ids]
        archetypes = [a for a in archetypes if a is not None]
        if not archetypes:
            return {}

        counts = Counter(archetypes)
        return {
            'n_ball_handlers':  counts.get(0, 0),   # Primary Ball-Handlers
            'n_3andD':          counts.get(1, 0),   # 3-and-D Wings
            'n_stretch_bigs':   counts.get(2, 0),   # Stretch Bigs
            'n_rim_runners':    counts.get(3, 0),   # Rim-Runners
            'n_creators':       counts.get(4, 0),   # Secondary Creators
            'n_twoway_wings':   counts.get(5, 0),   # Two-Way Wings
            'n_glue_guys':      counts.get(6, 0),   # Glue Guys
            'archetype_diversity': len(set(archetypes)),  # how many distinct roles
            'n_unique_archetypes': len(set(archetypes)),
        }

    features = df['player_ids_list'].apply(archetype_features)
    features_df = pd.DataFrame(features.tolist(), index=df.index)
    df = pd.concat([df, features_df], axis=1)
    df = df.drop(columns=['player_ids_list'])

    return df


# ── STEP 5: Archetype compatibility matrix ───────────────────────────────────

def archetype_compatibility_matrix(lineups_enriched: pd.DataFrame):
    """
    For 2-man lineups only, computes average synergy_delta for every
    pair of archetypes. Produces a heatmap — the compatibility matrix.
    """
    twos = lineups_enriched[
        (lineups_enriched['group_size'] == 2) &
        (lineups_enriched['archetype_fingerprint'].notna())
    ].copy()

    # Expand fingerprint into pair columns
    twos['a1'] = twos['archetype_fingerprint'].apply(lambda x: x[0] if x else None)
    twos['a2'] = twos['archetype_fingerprint'].apply(lambda x: x[1] if x else None)

    # Average synergy delta for each pair
    pair_deltas = {}
    for _, row in twos.iterrows():
        pair = (int(row['a1']), int(row['a2']))
        pair_deltas.setdefault(pair, []).append(row['synergy_delta'])

    # Build symmetric matrix
    matrix = np.full((N_CLUSTERS, N_CLUSTERS), np.nan)
    for (a, b), deltas in pair_deltas.items():
        if len(deltas) >= 5:   # need at least 5 pairs for stability
            avg = np.mean(deltas)
            matrix[a][b] = avg
            matrix[b][a] = avg

    labels = [ARCHETYPE_LABELS.get(i, str(i)) for i in range(N_CLUSTERS)]
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

    # Print
    print("\n── Archetype Compatibility Matrix (avg synergy_delta for 2-man lineups) ──")
    print(df_matrix.round(2).to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.isnan(matrix)
    sns.heatmap(
        df_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, mask=mask, linewidths=0.5, linecolor='#cccccc',
        ax=ax, cbar_kws={'label': 'Avg Synergy Delta'}
    )
    ax.set_title("Archetype Compatibility Matrix\n(2-man lineup synergy delta)",
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('archetype_compatibility_matrix.png', dpi=150)
    plt.close()
    print("  Saved archetype_compatibility_matrix.png")

    return df_matrix


# ── STEP 6: Best lineups per team by synergy delta ───────────────────────────

def best_lineups_per_team(lineups_enriched: pd.DataFrame, group_size: int = 5, top_n: int = 3):
    """
    For each team, print their top N lineups by synergy_delta.
    """
    df = lineups_enriched[lineups_enriched['group_size'] == group_size].copy()

    print(f"\n── Top {top_n} lineups (group_size={group_size}) per team by synergy_delta ──")
    teams = sorted(df['TEAM_ABBREVIATION'].dropna().unique()) if 'TEAM_ABBREVIATION' in df.columns else []

    results = []
    for team in teams:
        team_df = df[df['TEAM_ABBREVIATION'] == team].sort_values('synergy_delta', ascending=False).head(top_n)
        for rank, (_, row) in enumerate(team_df.iterrows(), 1):
            results.append({
                'team':              team,
                'rank':              rank,
                'lineup':            row['GROUP_NAME'],
                'min':               row['MIN'],
                'net_rating':        round(row['NET_RATING'], 1),
                'expected_net':      round(row['expected_net_rating'], 1),
                'synergy_delta':     round(row['synergy_delta'], 2),
                'archetype_combo':   row.get('archetype_fingerprint_str', ''),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('best_lineups_per_team.csv', index=False)
    print(f"  Saved best_lineups_per_team.csv ({len(results_df)} rows)")
    return results_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load
    players_df, lineups_df = load_data()

    # 2. Cluster players
    print("\nClustering players into archetypes...")
    players_with_archetypes, scaler, km = cluster_players(players_df)

    # 3. Visualise
    print("\nGenerating PCA visualisation...")
    plot_clusters(players_with_archetypes)

    # 4. Save player archetypes
    players_with_archetypes.to_csv('player_archetypes.csv', index=False)
    print(f"\n  Saved player_archetypes.csv ({len(players_with_archetypes)} players)")

    # 5. Enrich lineups
    print("\nEnriching lineups with archetype features...")
    archetype_lookup = build_archetype_lookup(players_with_archetypes)
    lineups_enriched = enrich_lineups(lineups_df, archetype_lookup)

    n_matched = lineups_enriched['archetype_fingerprint'].notna().sum()
    print(f"  {n_matched}/{len(lineups_enriched)} lineups matched to archetypes")

    lineups_enriched.to_csv('lineups_with_archetypes.csv', index=False)
    print(f"  Saved lineups_with_archetypes.csv")

    # 6. Compatibility matrix
    print("\nBuilding archetype compatibility matrix...")
    compatibility = archetype_compatibility_matrix(lineups_enriched)

    # 7. Best lineups per team
    best_lineups_per_team(lineups_enriched, group_size=5)
    best_lineups_per_team(lineups_enriched, group_size=2)

    # 8. Summary stats by archetype combo
    print("\n── Top 10 archetype combinations by avg synergy_delta (5-man) ──────")
    fives = lineups_enriched[
        (lineups_enriched['group_size'] == 5) &
        (lineups_enriched['archetype_fingerprint_str'].notna())
    ]
    combo_summary = (
        fives.groupby('archetype_fingerprint_str')['synergy_delta']
             .agg(mean='mean', std='std', count='count')
             .query('count >= 3')
             .sort_values('mean', ascending=False)
             .head(10)
    )
    # Decode fingerprint to labels for readability
    def decode(fp_str):
        try:
            ids = json.loads(fp_str.replace('(', '[').replace(')', ']'))
            return ' + '.join(ARCHETYPE_LABELS.get(i, str(i)) for i in ids)
        except Exception:
            return fp_str

    combo_summary['archetype_names'] = combo_summary.index.map(decode)
    print(combo_summary[['archetype_names', 'mean', 'std', 'count']].round(3).to_string())

    print("\nPhase 2 complete. Files saved:")
    print("  player_archetypes.csv")
    print("  lineups_with_archetypes.csv")
    print("  best_lineups_per_team.csv")
    print("  player_archetypes_pca.png")
    print("  archetype_compatibility_matrix.png")

    return players_with_archetypes, lineups_enriched, compatibility
