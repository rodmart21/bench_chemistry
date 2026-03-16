"""
NBA Lineup Chemistry — Enrich Lineups with UMAP+HDBSCAN Archetypes
===================================================================
Reads:
    player_archetypes_best.csv   (from phase2_umap_sweep.py)
    lineups_with_synergy.csv     (from phase1_synergy_delta.py)

Outputs:
    lineups_with_archetypes.csv
    archetype_compatibility_matrix.png

Run: python phase2_enrich_lineups.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────

ARCHETYPE_LABELS = {
    0: "Primary Ball-Handler",
    1: "Rim-Runner",
    2: "Star Versatile Engine",
    3: "3-and-D Wing",
    4: "Athletic Role Player",
    5: "Spot-Up Shooter",
}

PLAYERS_FILE             = 'results/umap/player_archetypes_best.csv'
LINEUPS_FILE             = 'results/synergy/lineups_with_synergy.csv'
ASSIGN_NOISE_TO_NEAREST  = True   # False = drop lineups with noise players instead


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_data(players_file=PLAYERS_FILE, lineups_file=LINEUPS_FILE):
    print("Loading data...")
    players = pd.read_csv(players_file)
    lineups = pd.read_csv(lineups_file)
    players['PLAYER_ID'] = players['PLAYER_ID'].astype(str)
    print(f"  {len(players)} players  |  {len(lineups)} lineups")
    print(f"  {(players['cluster'] == -1).sum()} noise players")
    return players, lineups


def assign_noise_players(players: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns noise players (cluster = -1) to their nearest cluster
    using KNN on the 2D UMAP coordinates.
    """
    noise_mask = players['cluster'] == -1
    n_noise    = noise_mask.sum()

    if n_noise == 0:
        print("  No noise players to assign.")
        return players

    print(f"\nAssigning {n_noise} noise players to nearest cluster via KNN...")
    df        = players.copy()
    umap_cols = ['umap_x', 'umap_y']
    non_noise = df[~noise_mask]
    noise     = df[noise_mask]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(non_noise[umap_cols], non_noise['cluster'])
    assigned = knn.predict(noise[umap_cols])

    df.loc[noise_mask, 'cluster'] = assigned

    for name, c in zip(noise['PLAYER_NAME'], assigned):
        print(f"  {name:30s} → {ARCHETYPE_LABELS.get(c, str(c))}")

    return df


def build_archetype_lookup(players: pd.DataFrame) -> dict:
    """Returns {player_id: cluster_int} for all players."""
    return dict(zip(players['PLAYER_ID'], players['cluster'].astype(int)))


def compute_fingerprint(player_ids: list, lookup: dict):
    """
    Sorted tuple of archetype ints for a lineup — e.g. (0, 1, 3, 3, 5).
    Returns None if any player is missing from the lookup.
    """
    archetypes = [lookup.get(pid) for pid in player_ids]
    if any(a is None for a in archetypes):
        return None
    return tuple(sorted(archetypes))


def compute_count_features(player_ids: list, lookup: dict) -> dict:
    """
    Returns one count column per archetype plus derived binary flags.
    """
    archetypes = [lookup.get(pid) for pid in player_ids]
    archetypes = [a for a in archetypes if a is not None]
    if not archetypes:
        return {}

    counts = Counter(archetypes)
    return {
        'n_ball_handlers':        counts.get(0, 0),
        'n_rim_runners':          counts.get(1, 0),
        'n_star_engines':         counts.get(2, 0),
        'n_3andD_wings':          counts.get(3, 0),
        'n_athletic_role':        counts.get(4, 0),
        'n_spotup_shooters':      counts.get(5, 0),
        'n_unique_archetypes':    len(set(archetypes)),
        'has_star_engine':        int(counts.get(2, 0) > 0),
        'has_rim_runner':         int(counts.get(1, 0) > 0),
        'has_ball_handler':       int(counts.get(0, 0) > 0),
        'multiple_ball_handlers': int(counts.get(0, 0) > 1),
        'multiple_rim_runners':   int(counts.get(1, 0) > 1),
    }


def enrich_lineups(lineups: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """
    Adds archetype fingerprint and count features to every lineup row.
    """
    print("\nEnriching lineups...")
    df = lineups.copy()

    df['player_ids_list'] = df['player_ids_str'].apply(
        lambda s: str(s).split(',') if pd.notna(s) else []
    )

    df['archetype_fingerprint'] = df['player_ids_list'].apply(
        lambda ids: compute_fingerprint(ids, lookup)
    )
    df['archetype_fingerprint_str'] = df['archetype_fingerprint'].apply(
        lambda x: str(x) if x is not None else None
    )
    df['archetype_fingerprint_named'] = df['archetype_fingerprint'].apply(
        lambda x: ' + '.join(ARCHETYPE_LABELS.get(i, str(i)) for i in x) if x else None
    )

    count_features = df['player_ids_list'].apply(
        lambda ids: compute_count_features(ids, lookup)
    )
    df = pd.concat([df, pd.DataFrame(count_features.tolist(), index=df.index)], axis=1)
    df = df.drop(columns=['player_ids_list'])

    n_matched = df['archetype_fingerprint'].notna().sum()
    print(f"  {n_matched}/{len(df)} lineups matched ({n_matched/len(df)*100:.1f}%)")
    return df


def build_compatibility_matrix(lineups_enriched: pd.DataFrame) -> pd.DataFrame:
    """Builds an archetype compatibility matrix based on avg synergy_delta for 2-man lineups."""
    
    N    = len(ARCHETYPE_LABELS)
    twos = lineups_enriched[
        (lineups_enriched['group_size'] == 2) &
        lineups_enriched['archetype_fingerprint'].notna()
    ].copy()

    twos['a1'] = twos['archetype_fingerprint'].apply(lambda x: x[0])
    twos['a2'] = twos['archetype_fingerprint'].apply(lambda x: x[1])

    # Compute league-wide prior from ALL 2-man synergy deltas
    prior_mean = twos['synergy_delta'].mean()
    prior_var  = twos['synergy_delta'].var()

    def bayesian_shrink(values):
        n        = len(values)
        obs_mean = np.mean(values)
        # Small n → pulled toward league average
        # Large n → stays close to observed mean
        weight = n / (n + prior_var)
        return weight * obs_mean + (1 - weight) * prior_mean

    pair_deltas = {}
    for _, row in twos.iterrows():
        pair = (int(row['a1']), int(row['a2']))
        pair_deltas.setdefault(pair, []).append(row['synergy_delta'])

    matrix = np.full((N, N), np.nan)
    for (a, b), deltas in pair_deltas.items():
        if len(deltas) >= 5:
            avg = bayesian_shrink(deltas)   # ← was np.mean(deltas)
            matrix[a][b] = avg
            matrix[b][a] = avg

    # rest of the function stays exactly the same...
    labels_list = [ARCHETYPE_LABELS[i] for i in range(N)]
    df_matrix   = pd.DataFrame(matrix, index=labels_list, columns=labels_list)

    print("\n── Archetype Compatibility Matrix (avg synergy_delta, 2-man) ──────────")
    print(df_matrix.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, mask=np.isnan(matrix),
        linewidths=0.5, linecolor='#cccccc',
        ax=ax, cbar_kws={'label': 'Avg Synergy Delta'}
    )
    ax.set_title("Archetype Compatibility Matrix\n(2-man lineup avg synergy delta)",
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('archetype_compatibility_matrix.png', dpi=150)
    plt.close()
    print("  Saved archetype_compatibility_matrix.png")

    return df_matrix


def print_top_combos(lineups_enriched: pd.DataFrame, group_size: int = 5, top_n: int = 10):
    """Prints top archetype combinations by avg synergy_delta for a given group size."""
    print(f"\n── Top {top_n} archetype combos by avg synergy_delta ({group_size}-man lineups) ──")
    subset = lineups_enriched[
        (lineups_enriched['group_size'] == group_size) &
        lineups_enriched['archetype_fingerprint_named'].notna()
    ]
    summary = (
        subset.groupby('archetype_fingerprint_named')['synergy_delta']
              .agg(mean='mean', std='std', count='count')
              .query('count >= 3')
              .sort_values('mean', ascending=False)
              .head(top_n)
    )
    print(summary.round(3).to_string())


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    players, lineups = load_data()

    if ASSIGN_NOISE_TO_NEAREST:
        players = assign_noise_players(players)

    lookup           = build_archetype_lookup(players)
    lineups_enriched = enrich_lineups(lineups, lookup)

    lineups_enriched.to_csv('lineups_with_archetypes.csv', index=False)
    print("  Saved lineups_with_archetypes.csv")

    compatibility = build_compatibility_matrix(lineups_enriched)

    print_top_combos(lineups_enriched, group_size=5)
    print_top_combos(lineups_enriched, group_size=2)

    print("\nDone.")
    return lineups_enriched, compatibility
