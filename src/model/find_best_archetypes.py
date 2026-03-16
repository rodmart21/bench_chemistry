"""
NBA Lineup Chemistry — Player Archetype Affinity
=================================================
Answers: "For any given player, which archetype brings the best out of them?"

Flavor 1 — Descriptive: rank archetypes by observed avg synergy delta per player
Flavor 2 — Predictive:  predict affinity for unseen player/archetype combinations

Uses ALL lineup group sizes (2, 3, 4, 5) — each player contributes one row
per lineup they appeared in, describing their synergy in the context of
the surrounding archetype composition.

Reads:
    lineups_with_archetypes.csv
    player_stats.csv
    player_archetypes_best.csv

Outputs:
    player_affinity_raw.csv          — one row per player x lineup
    player_affinity_profile.csv      — flavor 1: avg synergy per player x archetype
    player_best_archetype.csv        — flavor 1: best/worst archetype per player
    affinity_model.pkl               — flavor 2: trained model
    affinity_predictions.csv         — flavor 2: predicted affinity for all players x archetypes
    affinity_heatmap.png             — league-wide player x archetype heatmap
    top_players_per_archetype.png    — best players for each archetype partner

Run: python phase3_affinity.py
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────

ARCHETYPE_LABELS = {
    0: "Ball-Handler",
    1: "Rim-Runner",
    2: "Star Engine",
    3: "3-and-D Wing",
    4: "Athletic Role",
    5: "Spot-Up Shooter",
}

MIN_LINEUP_MINUTES  = 30
MIN_OBS_FOR_PROFILE = 3
TOP_N_PLAYERS       = 15

GBM_PARAMS = {
    'n_estimators':     300,
    'max_depth':        3,
    'learning_rate':    0.03,
    'subsample':        0.7,
    'min_samples_leaf': 15,
    'random_state':     42,
}

# Approximate avg TS_PCT per archetype (from your compatibility matrix)
ARCHETYPE_AVG_TS = {0: 0.573, 1: 0.635, 2: 0.585, 3: 0.507, 4: 0.571, 5: 0.594}


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    lineups    = pd.read_csv('results/umap/lineups_with_archetypes.csv')
    players    = pd.read_csv('results/player_stats.csv')
    archetypes = pd.read_csv('results/umap/player_archetypes_best.csv')

    players['PLAYER_ID']    = players['PLAYER_ID'].astype(str)
    archetypes['PLAYER_ID'] = archetypes['PLAYER_ID'].astype(str)

    players = players.merge(
        archetypes[['PLAYER_ID', 'cluster']],
        on='PLAYER_ID', how='left'
    )

    archetype_lookup = dict(zip(archetypes['PLAYER_ID'],
                                archetypes['cluster'].astype(int)))
    name_lookup      = dict(zip(players['PLAYER_ID'],
                                players['PLAYER_NAME']))

    stat_cols = ['USG_PCT', 'AST_PCT', 'TS_PCT', 'OREB_PCT', 'DREB_PCT',
                 'REB_PCT', 'PIE', 'NET_RATING', 'MIN', 'EFG_PCT']
    player_stats_lookup = {}
    for _, row in players.iterrows():
        pid = str(row['PLAYER_ID'])
        player_stats_lookup[pid] = {
            col: pd.to_numeric(row.get(col, np.nan), errors='coerce')
            for col in stat_cols
        }
        player_stats_lookup[pid]['cluster'] = archetype_lookup.get(pid, -1)
        player_stats_lookup[pid]['name']    = row.get('PLAYER_NAME', pid)

    print(f"  {len(lineups)} lineups  |  {len(players)} players")
    return lineups, players, archetype_lookup, name_lookup, player_stats_lookup


# ── BUILD RAW AFFINITY DATASET ────────────────────────────────────────────────

def build_affinity_dataset(lineups, archetype_lookup, name_lookup):
    """
    Expands every lineup into one row PER PLAYER in that lineup.
    Each row describes that player's synergy in the context of
    their partner archetype composition. Works for all group sizes 2-5.
    """
    print("\nBuilding player affinity dataset (all group sizes)...")
    rows = []
    df   = lineups[lineups['MIN'] >= MIN_LINEUP_MINUTES].copy()

    for _, lineup in df.iterrows():
        ids        = str(lineup['player_ids_str']).split(',')
        archetypes = [archetype_lookup.get(pid) for pid in ids]
        group_size = int(lineup['group_size'])

        if any(a is None for a in archetypes):
            continue

        for i, pid in enumerate(ids):
            partner_archetypes = [archetypes[j] for j in range(len(ids)) if j != i]
            partner_counts     = Counter(partner_archetypes)
            dominant           = max(partner_counts, key=partner_counts.get)

            rows.append({
                'player_id':                   pid,
                'player_name':                 name_lookup.get(pid, pid),
                'own_archetype':               archetypes[i],
                'own_archetype_name':          ARCHETYPE_LABELS.get(archetypes[i], str(archetypes[i])),
                'group_size':                  group_size,
                'n_ball_handler_partners':     partner_counts.get(0, 0),
                'n_rim_runner_partners':       partner_counts.get(1, 0),
                'n_star_engine_partners':      partner_counts.get(2, 0),
                'n_3andD_partners':            partner_counts.get(3, 0),
                'n_athletic_partners':         partner_counts.get(4, 0),
                'n_shooter_partners':          partner_counts.get(5, 0),
                'n_unique_partner_archetypes': len(set(partner_archetypes)),
                'dominant_partner_archetype':  dominant,
                'dominant_partner_name':       ARCHETYPE_LABELS.get(dominant, '?'),
                'synergy_delta':               float(lineup['synergy_delta'])
                                               if pd.notna(lineup['synergy_delta']) else np.nan,
                'off_synergy_delta':           float(lineup['off_synergy_delta'])
                                               if pd.notna(lineup.get('off_synergy_delta')) else np.nan,
                'def_synergy_delta':           float(lineup['def_synergy_delta'])
                                               if pd.notna(lineup.get('def_synergy_delta')) else np.nan,
                'NET_RATING':                  float(lineup['NET_RATING'])
                                               if pd.notna(lineup['NET_RATING']) else np.nan,
                'MIN':                         float(lineup['MIN']),
                'season':                      lineup.get('season', ''),
            })

    df_out = pd.DataFrame(rows).dropna(subset=['synergy_delta'])
    print(f"  {len(df_out)} player x lineup rows")
    print(f"  {df_out['player_id'].nunique()} unique players")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# FLAVOR 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════

def build_affinity_profile(df_affinity):
    """
    For each player x dominant partner archetype:
    compute minutes-weighted avg synergy delta.
    """
    print("\n── Flavor 1: Building affinity profiles ────────────────────────────")

    profile = (
        df_affinity
        .groupby(['player_id', 'player_name', 'own_archetype_name',
                  'dominant_partner_archetype', 'dominant_partner_name'])
        .apply(lambda g: pd.Series({
            'mean_synergy':     np.average(g['synergy_delta'], weights=g['MIN']),
            'mean_off_synergy': np.average(g['off_synergy_delta'].fillna(0), weights=g['MIN']),
            'mean_def_synergy': np.average(g['def_synergy_delta'].fillna(0), weights=g['MIN']),
            'total_min':        g['MIN'].sum(),
            'n_lineups':        len(g),
        }))
        .reset_index()
        .query(f'n_lineups >= {MIN_OBS_FOR_PROFILE}')
        .sort_values(['player_name', 'mean_synergy'], ascending=[True, False])
    )

    print(f"  {len(profile)} player x archetype pairs with {MIN_OBS_FOR_PROFILE}+ observations")
    return profile


def build_best_archetype_summary(profile):
    """
    For each player: best archetype, worst archetype, delta range, full ranking.
    """
    rows = []
    for player_id, group in profile.groupby('player_id'):
        ranked = group.sort_values('mean_synergy', ascending=False)
        best   = ranked.iloc[0]
        worst  = ranked.iloc[-1]
        ranking = dict(zip(
            ranked['dominant_partner_name'],
            ranked['mean_synergy'].round(3)
        ))
        rows.append({
            'player_id':           player_id,
            'player_name':         best['player_name'],
            'own_archetype':       best['own_archetype_name'],
            'best_partner':        best['dominant_partner_name'],
            'best_partner_delta':  round(best['mean_synergy'], 3),
            'worst_partner':       worst['dominant_partner_name'],
            'worst_partner_delta': round(worst['mean_synergy'], 3),
            'delta_range':         round(best['mean_synergy'] - worst['mean_synergy'], 3),
            'archetype_ranking':   str(ranking),
            'total_min':           int(group['total_min'].sum()),
        })

    df = pd.DataFrame(rows).sort_values('best_partner_delta', ascending=False)
    print(f"\n── Top 10 players by best partner synergy delta ────────────────────")
    print(df[['player_name', 'own_archetype', 'best_partner',
              'best_partner_delta', 'worst_partner', 'worst_partner_delta',
              'delta_range']].head(10).to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FLAVOR 2 — PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════

FLAVOR2_FEATURES = [
    'own_usg', 'own_ast', 'own_ts', 'own_oreb', 'own_dreb', 'own_net_rating',
    'own_archetype',
    'n_ball_handler_partners', 'n_rim_runner_partners', 'n_star_engine_partners',
    'n_3andD_partners', 'n_athletic_partners', 'n_shooter_partners',
    'n_unique_partner_archetypes',
    'dominant_partner_archetype',
    'group_size',
    'usg_compatibility',
    'ts_gap',
]


def add_player_stats_to_affinity(df_affinity, player_stats_lookup):
    """Joins each player's individual stats onto the affinity rows."""
    df = df_affinity.copy()

    for stat, col in [('USG_PCT',    'own_usg'),
                      ('AST_PCT',    'own_ast'),
                      ('TS_PCT',     'own_ts'),
                      ('OREB_PCT',   'own_oreb'),
                      ('DREB_PCT',   'own_dreb'),
                      ('NET_RATING', 'own_net_rating'),
                      ('cluster',    'own_archetype')]:
        df[col] = df['player_id'].apply(
            lambda pid: player_stats_lookup.get(pid, {}).get(stat, np.nan)
        )

    # Style compatibility: own high USG + Ball-Handler partner = crowded ball
    df['usg_compatibility'] = df.apply(
        lambda r: (r['own_usg'] or 0) * (1 if r['dominant_partner_archetype'] != 0 else -1)
        if pd.notna(r.get('own_usg')) else np.nan, axis=1
    )
    # TS gap: own shooting vs avg TS of partner archetype
    df['ts_gap'] = df.apply(
        lambda r: (r.get('own_ts') or np.nan) -
                  ARCHETYPE_AVG_TS.get(r['dominant_partner_archetype'], 0.55),
        axis=1
    )
    return df


def train_affinity_model(df_enriched):
    """Trains GBM to predict synergy_delta from player stats + partner context."""
    print("\n── Flavor 2: Training affinity prediction model ─────────────────────")

    available = [f for f in FLAVOR2_FEATURES if f in df_enriched.columns]
    df_clean  = df_enriched[available + ['synergy_delta']].dropna()
    X, y      = df_clean[available], df_clean['synergy_delta']

    print(f"  Training on {len(X)} rows  |  {len(available)} features")

    model  = GradientBoostingRegressor(**GBM_PARAMS)
    cv     = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_cv  = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_cv = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')

    print(f"  CV R²:  {r2_cv.mean():.3f} ± {r2_cv.std():.3f}")
    print(f"  CV MAE: {(-mae_cv).mean():.3f} ± {(-mae_cv).std():.3f}")

    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"  Train R²:  {r2_score(y, y_pred):.3f}")
    print(f"  Train MAE: {mean_absolute_error(y, y_pred):.3f}")

    importance = pd.Series(model.feature_importances_, index=available)
    print(f"\n  Top features:")
    print(importance.sort_values(ascending=False).head(8).round(4).to_string())

    return model, available


def predict_all_affinities(players_df, model, feature_names, player_stats_lookup):
    """
    For every player x every archetype, predict synergy delta.
    Fills in gaps where no observed data exists.
    """
    print("\n  Predicting affinities for all player x archetype combinations...")
    rows = []

    for _, player_row in players_df.iterrows():
        pid    = str(player_row['PLAYER_ID'])
        pstats = player_stats_lookup.get(pid, {})
        if not pstats:
            continue
        own_archetype = pstats.get('cluster', -1)
        if own_archetype == -1:
            continue

        for partner_archetype in range(len(ARCHETYPE_LABELS)):
            feats = {
                'own_usg':                     pstats.get('USG_PCT', 0) or 0,
                'own_ast':                     pstats.get('AST_PCT', 0) or 0,
                'own_ts':                      pstats.get('TS_PCT',  0) or 0,
                'own_oreb':                    pstats.get('OREB_PCT', 0) or 0,
                'own_dreb':                    pstats.get('DREB_PCT', 0) or 0,
                'own_net_rating':              pstats.get('NET_RATING', 0) or 0,
                'own_archetype':               own_archetype,
                'group_size':                  2,
                'n_ball_handler_partners':     1 if partner_archetype == 0 else 0,
                'n_rim_runner_partners':       1 if partner_archetype == 1 else 0,
                'n_star_engine_partners':      1 if partner_archetype == 2 else 0,
                'n_3andD_partners':            1 if partner_archetype == 3 else 0,
                'n_athletic_partners':         1 if partner_archetype == 4 else 0,
                'n_shooter_partners':          1 if partner_archetype == 5 else 0,
                'n_unique_partner_archetypes': 1,
                'dominant_partner_archetype':  partner_archetype,
                'usg_compatibility':           (pstats.get('USG_PCT', 0) or 0) *
                                               (1 if partner_archetype != 0 else -1),
                'ts_gap':                      (pstats.get('TS_PCT', 0) or 0) -
                                               ARCHETYPE_AVG_TS.get(partner_archetype, 0.55),
            }

            X    = pd.DataFrame([[feats.get(f, 0) for f in feature_names]], columns=feature_names)
            pred = model.predict(X)[0]

            rows.append({
                'player_id':              pid,
                'player_name':            pstats.get('name', pid),
                'own_archetype':          own_archetype,
                'own_archetype_name':     ARCHETYPE_LABELS.get(own_archetype, str(own_archetype)),
                'partner_archetype':      partner_archetype,
                'partner_archetype_name': ARCHETYPE_LABELS.get(partner_archetype, str(partner_archetype)),
                'predicted_synergy':      round(pred, 3),
            })

    df_preds = pd.DataFrame(rows)
    best_idx = df_preds.groupby('player_id')['predicted_synergy'].idxmax()
    df_preds['is_best'] = False
    df_preds.loc[best_idx, 'is_best'] = True
    return df_preds


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_affinity_heatmap(predictions_df, top_n=40):
    top_players = (
        predictions_df.groupby('player_id')['predicted_synergy']
                      .max().nlargest(top_n).index
    )
    pivot = (
        predictions_df[predictions_df['player_id'].isin(top_players)]
        .pivot_table(index='player_name', columns='partner_archetype_name',
                     values='predicted_synergy', aggfunc='mean')
    )
    pivot['_max'] = pivot.max(axis=1)
    pivot = pivot.sort_values('_max', ascending=False).drop(columns='_max')

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.4, linecolor='#dddddd', ax=ax,
                cbar_kws={'label': 'Predicted Synergy Delta'})
    ax.set_title(f"Predicted Player × Archetype Affinity (top {top_n})",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Partner Archetype"); ax.set_ylabel("")
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('affinity_heatmap.png', dpi=150)
    plt.close()
    print("  Saved affinity_heatmap.png")


def plot_top_players_per_archetype(predictions_df, top_n=TOP_N_PLAYERS):
    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    axes      = axes.flatten()

    for idx, (arch_id, arch_name) in enumerate(ARCHETYPE_LABELS.items()):
        ax   = axes[idx]
        df_a = (predictions_df[predictions_df['partner_archetype'] == arch_id]
                .sort_values('predicted_synergy', ascending=False)
                .head(top_n))
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in df_a['predicted_synergy']]
        ax.barh(df_a['player_name'], df_a['predicted_synergy'],
                color=colors, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(f"Best players to pair with\n{arch_name}", fontweight='bold', fontsize=10)
        ax.set_xlabel("Predicted Synergy Delta")
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle("Top Players by Partner Archetype Affinity",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('top_players_per_archetype.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved top_players_per_archetype.png")


# ── QUERY ─────────────────────────────────────────────────────────────────────

def query_player(player_name, summary_df, predictions_df):
    """Prints observed + predicted archetype ranking for any player."""
    print(f"\n{'='*60}")
    print(f"  AFFINITY REPORT: {player_name.upper()}")
    print(f"{'='*60}")

    obs = summary_df[summary_df['player_name'].str.contains(player_name, case=False, na=False)]
    if not obs.empty:
        row = obs.iloc[0]
        print(f"\n  [Observed]  Own archetype: {row['own_archetype']}")
        print(f"  Best partner:  {row['best_partner']}  ({row['best_partner_delta']:+.3f})")
        print(f"  Worst partner: {row['worst_partner']}  ({row['worst_partner_delta']:+.3f})")
        print(f"  Full ranking:  {row['archetype_ranking']}")
    else:
        print("  No observed data found.")

    pred = predictions_df[
        predictions_df['player_name'].str.contains(player_name, case=False, na=False)
    ].sort_values('predicted_synergy', ascending=False)
    if not pred.empty:
        print(f"\n  [Predicted] Archetype affinity ranking:")
        for _, r in pred.iterrows():
            bar  = '█' * max(0, int((r['predicted_synergy'] + 2) * 3))
            flag = ' ← BEST' if r['is_best'] else ''
            print(f"  {r['partner_archetype_name']:20s}  {r['predicted_synergy']:+.3f}  {bar}{flag}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    lineups, players, archetype_lookup, name_lookup, player_stats_lookup = load_data()

    # Build raw dataset
    df_affinity = build_affinity_dataset(lineups, archetype_lookup, name_lookup)
    df_affinity.to_csv('player_affinity_raw.csv', index=False)
    print("  Saved player_affinity_raw.csv")

    # Flavor 1
    profile = build_affinity_profile(df_affinity)
    profile.to_csv('player_affinity_profile.csv', index=False)
    print("  Saved player_affinity_profile.csv")

    summary = build_best_archetype_summary(profile)
    summary.to_csv('player_best_archetype.csv', index=False)
    print("  Saved player_best_archetype.csv")

    # Flavor 2
    df_enriched = add_player_stats_to_affinity(df_affinity, player_stats_lookup)
    model, feature_names = train_affinity_model(df_enriched)

    with open('affinity_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_names}, f)
    print("  Saved affinity_model.pkl")

    predictions = predict_all_affinities(players, model, feature_names, player_stats_lookup)
    predictions.to_csv('affinity_predictions.csv', index=False)
    print("  Saved affinity_predictions.csv")

    # Plots
    print("\nGenerating plots...")
    plot_affinity_heatmap(predictions)
    plot_top_players_per_archetype(predictions)

    # Example queries — edit these names to explore any player
    for name in ["Jokić", "Gilgeous-Alexander", "Dončić"]:
        query_player(name, summary, predictions)

    print("\nDone.")
    return df_affinity, profile, summary, model, predictions


if __name__ == '__main__':
    df_affinity, profile, summary, model, predictions = main()