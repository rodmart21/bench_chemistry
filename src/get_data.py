"""
NBA Lineup Chemistry — Phase 1: Synergy Delta
==============================================
Computes the difference between a lineup's actual NET_RATING
and the expected NET_RATING based on individual player quality.

Positive delta  →  group creates emergent value (chemistry)
Negative delta  →  group underperforms individual talent (anti-chemistry)

Requirements:
    pip install nba_api pandas numpy
"""

import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashlineups, leaguedashplayerstats, teamdashlineups
from nba_api.stats.static import teams


# ── CONFIG ────────────────────────────────────────────────────────────────────

SEASONS       = ['2025-26']          # add more e.g. ['2022-23', '2023-24', '2024-25', '2025-26']
GROUP_SIZES   = [2, 3, 4, 5]
MIN_THRESHOLDS = {2: 100, 3: 75, 4: 50, 5: 30}   # minimum shared minutes to include a lineup


# ── STEP 1: Pull individual player stats ─────────────────────────────────────

def get_player_stats(season: str) -> pd.DataFrame:
    """
    Fetches individual advanced stats for all players in a season.
    Returns a DataFrame indexed by PLAYER_ID (as string).
    """
    print(f"  Fetching individual player stats for {season}...")
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame',
    ).get_data_frames()[0]

    # Keep only what we need
    cols = [
        'PLAYER_ID', 'PLAYER_NAME',
        'NET_RATING', 'OFF_RATING', 'DEF_RATING',
        'PIE', 'TS_PCT', 'USG_PCT',
        'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
        'MIN', 'GP'
    ]
    df = df[cols].copy()
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(str)
    df['season'] = season
    return df


# ── STEP 2: Pull lineup data ──────────────────────────────────────────────────

def get_lineups(season: str, group_size: int) -> pd.DataFrame:
    """
    Fetches all lineup combinations of a given group size for a season.
    Iterates through all teams to bypass the 2000 row API limit on LeagueDashLineups.
    """
    print(f"  Fetching {group_size}-man lineups for {season}...")
    
    nba_teams = teams.get_teams()
    team_dfs = []
    
    for team in nba_teams:
        team_id = team['id']
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                team_df = teamdashlineups.TeamDashLineups(
                    team_id=team_id,
                    season=season,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame',
                    group_quantity=group_size,
                    timeout=60
                ).get_data_frames()[1]  # index 1 usually has the lineup data for TeamDashLineups
                team_dfs.append(team_df)
                success = True
                time.sleep(1)  # respect NBA API rate limits
            except Exception as e:
                print(f"    Timeout/error for team {team_id}, retries left: {retries - 1}")
                retries -= 1
                time.sleep(5)  # Backoff before retrying
                if retries == 0:
                    print(f"    Failed to fetch team {team_id}")
        
    if not team_dfs:
        return pd.DataFrame()
        
    df = pd.concat(team_dfs, ignore_index=True)

    df['season'] = season
    df['group_size'] = group_size
    return df


# ── STEP 3: Parse player IDs from GROUP_ID ───────────────────────────────────

def parse_player_ids(group_id: str) -> list[str]:
    """
    GROUP_ID format: '-201142-1641708-203076-'
    Returns: ['201142', '1641708', '203076']
    """
    return [pid for pid in group_id.strip('-').split('-') if pid]


# ── STEP 4: Compute expected NET_RATING for a lineup ─────────────────────────

def compute_expected_net(player_ids: list[str], player_lookup: dict) -> float | None:
    """
    Expected NET_RATING = simple average of each player's individual NET_RATING.
    Returns None if any player is missing from the lookup (insufficient data).
    """
    ratings = []
    for pid in player_ids:
        rating = player_lookup.get(pid)
        if rating is None or pd.isna(rating):
            return None
        ratings.append(rating)
    return np.mean(ratings)


def compute_expected_metric(player_ids: list[str], player_lookup: dict) -> float | None:
    """Generic version of compute_expected_net for any metric."""
    vals = [player_lookup.get(pid) for pid in player_ids]
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    return np.mean(vals) if len(vals) == len(player_ids) else None


# ── STEP 5: Compute USG-weighted TS_PCT ──────────────────────────────────────

def compute_weighted_ts(player_ids: list[str], ts_lookup: dict, usg_lookup: dict) -> float | None:
    """
    TS_PCT weighted by USG_PCT — gives more weight to players who actually shoot more.
    This is a better expected shooting efficiency than a simple average.
    """
    ts_vals  = [ts_lookup.get(pid)  for pid in player_ids]
    usg_vals = [usg_lookup.get(pid) for pid in player_ids]

    if any(v is None or pd.isna(v) for v in ts_vals + usg_vals):
        return None

    total_usg = sum(usg_vals)
    if total_usg == 0:
        return None

    return sum(ts * usg for ts, usg in zip(ts_vals, usg_vals)) / total_usg


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    all_lineups = []
    all_players = []

    for season in SEASONS:
        print(f"\n{'='*50}")
        print(f"Season: {season}")
        print(f"{'='*50}")

        # 1. Individual player stats
        player_df = get_player_stats(season)
        all_players.append(player_df)
        time.sleep(1)

        # 2. Build lookup dicts: player_id → metric
        player_net_lookup = dict(zip(player_df['PLAYER_ID'], player_df['NET_RATING']))
        player_off_lookup = dict(zip(player_df['PLAYER_ID'], player_df['OFF_RATING']))
        player_def_lookup = dict(zip(player_df['PLAYER_ID'], player_df['DEF_RATING']))
        player_pie_lookup = dict(zip(player_df['PLAYER_ID'], player_df['PIE']))
        player_ts_lookup  = dict(zip(player_df['PLAYER_ID'], player_df['TS_PCT']))
        player_usg_lookup = dict(zip(player_df['PLAYER_ID'], player_df['USG_PCT']))

        # 3. Lineup data for each group size
        for group_size in GROUP_SIZES:
            df = get_lineups(season, group_size)

            # Parse player IDs
            df['player_ids'] = df['GROUP_ID'].apply(parse_player_ids)

            # Apply minimum minutes filter
            min_thresh = MIN_THRESHOLDS[group_size]
            df = df[df['MIN'] >= min_thresh].copy()
            print(f"    {len(df)} lineups after {min_thresh}min filter")

            # Compute expected values from individual stats
            df['expected_net_rating'] = df['player_ids'].apply(
                lambda ids: compute_expected_net(ids, player_net_lookup)
            )
            df['expected_off_rating'] = df['player_ids'].apply(
                lambda ids: compute_expected_metric(ids, player_off_lookup)
            )
            df['expected_def_rating'] = df['player_ids'].apply(
                lambda ids: compute_expected_metric(ids, player_def_lookup)
            )
            df['expected_pie'] = df['player_ids'].apply(
                lambda ids: compute_expected_metric(ids, player_pie_lookup)
            )
            df['expected_ts_usg_weighted'] = df['player_ids'].apply(
                lambda ids: compute_weighted_ts(ids, player_ts_lookup, player_usg_lookup)
            )

            # Compute synergy deltas
            df['synergy_delta']     = df['NET_RATING'] - df['expected_net_rating']
            df['off_synergy_delta'] = df['OFF_RATING'] - df['expected_off_rating']
            df['def_synergy_delta'] = df['DEF_RATING'] - df['expected_def_rating']   # lower is better
            df['pie_synergy_delta'] = df['PIE']        - df['expected_pie']

            # Convert player_ids list → string for CSV export
            df['player_ids_str'] = df['player_ids'].apply(lambda x: ','.join(x))
            df = df.drop(columns=['player_ids'])

            all_lineups.append(df)

    # ── Combine and save ──────────────────────────────────────────────────────

    lineups_df = pd.concat(all_lineups, ignore_index=True)
    players_df = pd.concat(all_players, ignore_index=True)

    # Drop rows where expected value couldn't be computed
    lineups_df = lineups_df.dropna(subset=['synergy_delta'])

    # Save
    lineups_df.to_csv('lineups_with_synergy.csv', index=False)
    players_df.to_csv('player_stats.csv', index=False)
    print(f"\nSaved {len(lineups_df)} lineups to lineups_with_synergy.csv")

    # ── Quick sanity check ────────────────────────────────────────────────────

    print("\n" + "="*50)
    print("TOP 10 LINEUPS BY SYNERGY DELTA (any group size)")
    print("="*50)
    top = (
        lineups_df[['GROUP_NAME', 'TEAM_ABBREVIATION', 'group_size', 'MIN',
                    'NET_RATING', 'expected_net_rating', 'synergy_delta', 'season']]
        .sort_values('synergy_delta', ascending=False)
        .head(10)
    )
    print(top.to_string(index=False))

    print("\n" + "="*50)
    print("WORST 10 LINEUPS BY SYNERGY DELTA (anti-chemistry)")
    print("="*50)
    bottom = (
        lineups_df[['GROUP_NAME', 'TEAM_ABBREVIATION', 'group_size', 'MIN',
                    'NET_RATING', 'expected_net_rating', 'synergy_delta', 'season']]
        .sort_values('synergy_delta', ascending=True)
        .head(10)
    )
    print(bottom.to_string(index=False))

    print("\n" + "="*50)
    print("AVERAGE SYNERGY DELTA BY GROUP SIZE")
    print("="*50)
    summary = lineups_df.groupby('group_size')['synergy_delta'].agg(['mean', 'std', 'count'])
    print(summary.round(3))

    return lineups_df, players_df
