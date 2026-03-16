"""
NBA Lineup Chemistry — Phase 3A: Gradient Boosting Synergy Predictor
=====================================================================
Trains a GBM to predict synergy_delta for any lineup combination.
After training, you can pass in any group of players (even ones who
have never shared the court) and get a predicted synergy_delta.

Reads:
    lineups_with_archetypes.csv   (from phase2_enrich_lineups.py)
    player_stats.csv              (from phase1_synergy_delta.py)
    player_archetypes_best.csv    (from phase2_umap_sweep.py)

Outputs:
    gbm_model.pkl                 — trained model
    feature_importance.png        — what drives chemistry
    model_evaluation.png          — actual vs predicted + residuals
    phase3_predictions.csv        — predictions for any lineups you define

Run: python phase3_gbm.py
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from collections import Counter

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

# GBM hyperparameters — tune these if needed
GBM_PARAMS = {
    'n_estimators':     300,
    'max_depth':        3,        # was 4 — shallower trees generalize better
    'learning_rate':    0.03,     # was 0.05 — slower learning = less overfit
    'subsample':        0.7,      # was 0.8
    'min_samples_leaf': 20,       # was 10 — requires more support per leaf
    'random_state':     42,
}

# Features used for training — edit to experiment
LINEUP_FEATURES = [
    # Archetype composition counts
    'n_ball_handlers', 'n_rim_runners', 'n_star_engines',
    'n_3andD_wings', 'n_athletic_role', 'n_spotup_shooters',
    'n_unique_archetypes',
    # Derived flags
    'has_star_engine', 'has_rim_runner', 'has_ball_handler',
    'multiple_ball_handlers', 'multiple_rim_runners',
    # Group size
    'group_size',
    # Collective style features (engineered below)
    'mean_usg',        'std_usg',         # usage spread — do players share the ball?
    'mean_ast_pct',    'max_ast_pct',      # playmaking presence
    'mean_ts',         'usg_weighted_ts',  # shooting quality
    'mean_oreb',       'mean_dreb',        # rebounding profile
    'usg_gini',                            # inequality of usage (0=equal, 1=one player hogs all)
    # 'expected_net_rating',                 # individual talent baseline
]

TARGET = 'synergy_delta'
MIN_MINUTES = 30    # minimum shared minutes to include a lineup in training


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    lineups = pd.read_csv('results/umap/lineups_with_archetypes.csv')
    players = pd.read_csv('results/player_stats.csv')
    archetypes = pd.read_csv('results/umap/player_archetypes_best.csv')

    players['PLAYER_ID']    = players['PLAYER_ID'].astype(str)
    archetypes['PLAYER_ID'] = archetypes['PLAYER_ID'].astype(str)

    # Merge cluster onto player stats
    players = players.merge(
        archetypes[['PLAYER_ID', 'cluster']],
        on='PLAYER_ID', how='left'
    )

    print(f"  {len(lineups)} lineups  |  {len(players)} players")
    return lineups, players


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def gini(values):
    """Gini coefficient — measures inequality of usage across players."""
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1)) * values) - (n + 1) * values.sum()) / (n * values.sum())


def engineer_group_features(player_ids: list, player_lookup: dict) -> dict | None:
    rows = [player_lookup.get(pid) for pid in player_ids]
    if any(r is None for r in rows):
        return None

    def safe_get(row, key):
        val = row.get(key, np.nan)
        return 0.0 if pd.isna(val) else float(val)   # ← replace NaN with 0

    usg  = [safe_get(r, 'USG_PCT')    for r in rows]
    ast  = [safe_get(r, 'AST_PCT')    for r in rows]
    ts   = [safe_get(r, 'TS_PCT')     for r in rows]
    oreb = [safe_get(r, 'OREB_PCT')   for r in rows]
    dreb = [safe_get(r, 'DREB_PCT')   for r in rows]

    total_usg = sum(usg)
    usg_weighted_ts = (
        sum(t * u for t, u in zip(ts, usg)) / total_usg
        if total_usg > 0 else np.mean(ts)
    )

    return {
        'mean_usg':         np.mean(usg),
        'std_usg':          np.std(usg),
        'mean_ast_pct':     np.mean(ast),
        'max_ast_pct':      np.max(ast),
        'mean_ts':          np.mean(ts),
        'usg_weighted_ts':  usg_weighted_ts,
        'mean_oreb':        np.mean(oreb),
        'mean_dreb':        np.mean(dreb),
        'usg_gini':         gini(usg),
    }


def build_player_lookup(players: pd.DataFrame) -> dict:
    """
    Returns {player_id: {stat: value}} for fast feature engineering.
    """
    stat_cols = ['USG_PCT', 'AST_PCT', 'TS_PCT', 'OREB_PCT', 'DREB_PCT',
                 'PACE', 'TM_TOV_PCT', 'EFG_PCT', 'NET_RATING', 'MIN']
    lookup = {}
    for _, row in players.iterrows():
        pid = str(row['PLAYER_ID'])
        lookup[pid] = {col: pd.to_numeric(row.get(col, np.nan), errors='coerce')
                       for col in stat_cols}
    return lookup


def prepare_features(lineups: pd.DataFrame, player_lookup: dict) -> pd.DataFrame:
    """
    Engineers all group-level features and returns a clean feature matrix.
    """
    print("\nEngineering features...")
    df = lineups.copy()
    df = df[df['MIN'] >= MIN_MINUTES].copy()
    df = df[df[TARGET].notna()].copy()

    # Parse player IDs
    df['player_ids_list'] = df['player_ids_str'].apply(
        lambda s: str(s).split(',') if pd.notna(s) else []
    )

    # Engineer group features
    group_feats = df['player_ids_list'].apply(
        lambda ids: engineer_group_features(ids, player_lookup)
    )
    group_df = pd.DataFrame(group_feats.tolist(), index=df.index)
    df = pd.concat([df, group_df], axis=1)

    # Drop rows where feature engineering failed (missing players)
    df = df.dropna(subset=['mean_usg'])

    # Ensure archetype count columns are numeric
    archetype_cols = ['n_ball_handlers', 'n_rim_runners', 'n_star_engines',
                      'n_3andD_wings', 'n_athletic_role', 'n_spotup_shooters',
                      'n_unique_archetypes', 'has_star_engine', 'has_rim_runner',
                      'has_ball_handler', 'multiple_ball_handlers', 'multiple_rim_runners']
    for col in archetype_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['group_size']           = pd.to_numeric(df['group_size'], errors='coerce')
    df['expected_net_rating']  = pd.to_numeric(df['expected_net_rating'], errors='coerce')

    print(f"  {len(df)} lineups after filtering  |  {len(LINEUP_FEATURES)} features")
    return df


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Trains GBM with cross-validation and returns the fitted model + scores.
    """
    print("\nTraining Gradient Boosting model...")

    available_features = [f for f in LINEUP_FEATURES if f in df.columns]
    X = df[available_features].fillna(0)
    y = df[TARGET].astype(float)

    model = GradientBoostingRegressor(**GBM_PARAMS)

    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores  = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')

    print(f"  Cross-validation R²:  {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
    print(f"  Cross-validation MAE: {(-mae_scores).mean():.3f} ± {(-mae_scores).std():.3f}")
    print(f"  (R² of 0 = predicting the mean, 1.0 = perfect)")

    # Fit on full data for final model
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"  Train R²:  {r2_score(y, y_pred):.3f}")
    print(f"  Train MAE: {mean_absolute_error(y, y_pred):.3f}")

    return model, X, y, y_pred, available_features


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2ecc71' if v > 0.05 else '#95a5a6' for v in importance.values]
    importance.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.axvline(0.05, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='5% threshold')
    ax.set_title("Feature Importance — What Drives Lineup Chemistry?",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("  Saved feature_importance.png")

    print("\n── Top 10 most important features ──────────────────────────────────")
    print(importance.sort_values(ascending=False).head(10).round(4).to_string())


def plot_evaluation(y_true, y_pred):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Actual vs predicted
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, color='steelblue', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax1.set_xlabel("Actual Synergy Delta")
    ax1.set_ylabel("Predicted Synergy Delta")
    ax1.set_title("Actual vs Predicted", fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.2)

    # Residuals
    residuals = y_true - y_pred
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='red', linewidth=1.5, linestyle='--')
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight='bold')
    ax2.grid(alpha=0.2)

    plt.suptitle(f"Model Evaluation  |  R²={r2_score(y_true, y_pred):.3f}  "
                 f"MAE={mean_absolute_error(y_true, y_pred):.3f}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150)
    plt.close()
    print("  Saved model_evaluation.png")


# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────

def predict_lineup(player_ids: list,
                   model,
                   player_lookup: dict,
                   archetype_lookup: dict,
                   feature_names: list) -> dict:
    """
    Given a list of player IDs, predicts the expected synergy_delta.
    Works for 2, 3, 4, or 5 players — even ones who never played together.

    Returns a dict with prediction + confidence info + feature breakdown.
    """
    player_ids = [str(pid) for pid in player_ids]
    group_size = len(player_ids)

    # Group style features
    group_feats = engineer_group_features(player_ids, player_lookup)
    if group_feats is None:
        missing = [pid for pid in player_ids if pid not in player_lookup]
        return {"error": f"Missing players: {missing}"}

    # Archetype features
    archetypes = [archetype_lookup.get(pid) for pid in player_ids]
    archetypes = [a for a in archetypes if a is not None]
    counts     = Counter(archetypes)

    archetype_feats = {
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
        'group_size':             group_size,
    }

    # Expected net rating baseline
    net_ratings = [player_lookup[pid].get('NET_RATING', 0) for pid in player_ids]
    minutes     = [player_lookup[pid].get('MIN', 1) for pid in player_ids]
    total_min   = sum(minutes)
    expected_net = sum(r * m for r, m in zip(net_ratings, minutes)) / total_min

    all_feats = {**group_feats, **archetype_feats, 'expected_net_rating': expected_net}

    # Build feature vector in correct order
    X = pd.DataFrame([{f: all_feats.get(f, 0) for f in feature_names}])
    X = X.fillna(0)     
    pred = model.predict(X)[0]

    archetype_names = [ARCHETYPE_LABELS.get(a, str(a)) for a in archetypes]

    return {
        "predicted_synergy_delta": round(pred, 3),
        "expected_net_rating":     round(expected_net, 2),
        "predicted_net_rating":    round(expected_net + pred, 2),
        "group_size":              group_size,
        "archetypes":              archetype_names,
        "archetype_fingerprint":   str(tuple(sorted(archetypes))),
        "features":                {k: round(v, 4) for k, v in all_feats.items()},
    }


def predict_by_name(player_names: list,
                    model,
                    player_lookup: dict,
                    archetype_lookup: dict,
                    feature_names: list,
                    name_to_id: dict) -> dict:
    """Wrapper to predict by player name instead of ID."""
    player_ids = []
    missing    = []
    for name in player_names:
        pid = name_to_id.get(name)
        if pid is None:
            # fuzzy match — find closest name
            matches = [n for n in name_to_id if name.lower() in n.lower()]
            if matches:
                pid = name_to_id[matches[0]]
                print(f"  '{name}' matched to '{matches[0]}'")
            else:
                missing.append(name)
        if pid:
            player_ids.append(pid)

    if missing:
        print(f"  Could not find: {missing}")

    return predict_lineup(player_ids, model, player_lookup, archetype_lookup, feature_names)


# ── SAVE / LOAD MODEL ─────────────────────────────────────────────────────────

def save_model(model, feature_names, path='gbm_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_names}, f)
    print(f"  Saved {path}")


def load_model(path='gbm_model.pkl'):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['model'], obj['features']


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load
    lineups, players = load_data()

    player_lookup    = build_player_lookup(players)
    archetype_lookup = dict(zip(players['PLAYER_ID'], players['cluster'].fillna(-1).astype(int)))
    name_to_id       = dict(zip(players['PLAYER_NAME'], players['PLAYER_ID'].astype(str)))

    # 2. Feature engineering
    df_features = prepare_features(lineups, player_lookup)

    # 3. Train
    model, X, y, y_pred, feature_names = train_model(df_features)

    # 4. Plots
    print("\nGenerating plots...")
    plot_feature_importance(model, feature_names)
    plot_evaluation(y, y_pred)

    # 5. Save model
    save_model(model, feature_names)

    # ── EXAMPLE PREDICTIONS ───────────────────────────────────────────────────
    # Edit these player names to predict any combination you want
    example_lineups = [
        ["Nikola Jokić", "Jamal Murray", "Michael Porter Jr.", "Aaron Gordon", "Kentavious Caldwell-Pope"],
        ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Isaiah Hartenstein", "Lu Dort"],
        ["Luka Dončić", "Kyrie Irving", "P.J. Washington", "Derrick Jones Jr.", "Daniel Gafford"],
    ]

    print("\n" + "="*65)
    print("EXAMPLE PREDICTIONS")
    print("="*65)

    predictions = []
    for names in example_lineups:
        result = predict_by_name(names, model, player_lookup,
                                 archetype_lookup, feature_names, name_to_id)
        print(f"\n  Lineup: {', '.join(names)}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Archetypes:               {result['archetypes']}")
            print(f"  Expected NET_RATING:      {result['expected_net_rating']}")
            print(f"  Predicted synergy delta:  {result['predicted_synergy_delta']:+.3f}")
            print(f"  Predicted NET_RATING:     {result['predicted_net_rating']}")
            predictions.append({
                'lineup': ', '.join(names),
                **{k: v for k, v in result.items() if k != 'features'}
            })

    if predictions:
        pd.DataFrame(predictions).to_csv('phase3_predictions.csv', index=False)
        print("\n  Saved phase3_predictions.csv")

    print("\nDone.")
    return model, player_lookup, archetype_lookup, feature_names, name_to_id


if __name__ == '__main__':
    model, player_lookup, archetype_lookup, feature_names, name_to_id = main()