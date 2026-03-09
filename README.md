# "Bench Chemistry" Synergy Analysis — Data Dictionary

This document describes every variable used in the NBA lineup performance datasets. These metrics evaluate how specific combinations of players perform collectively on the court.

---

## Lineup Identifiers

| Variable | Full Name | Description |
|---|---|---|
| GROUP_SET | Group Set | The category of the lineup (e.g., "5-man Lineups", "2-man Lineups"). |
| GROUP_ID | Group Identifier | A unique numerical ID assigned by the NBA to this specific combination of players. |
| GROUP_NAME | Group Name | The names of the players included in the lineup, typically separated by hyphens. |

---

## Volume & Success Metrics

| Variable | Full Name | Description |
|---|---|---|
| GP | Games Played | The number of games in which this specific lineup appeared together. |
| W | Wins | The number of games won by the team when this lineup played. |
| L | Losses | The number of games lost by the team when this lineup played. |
| W_PCT | Win Percentage | The percentage of games won (W / GP). |
| MIN | Minutes Played | Total minutes the specific group has played together on the floor. |
| SUM_TIME_PLAYED | Summary Time | The cumulative total of individual minutes for all players in the group. |

---

## Efficiency & Net Ratings

> Ratings are calculated **per 100 possessions** to standardize performance across different game speeds and paces.

| Variable | Full Name | Description |
|---|---|---|
| OFF_RATING | Offensive Rating | Points scored by the lineup per 100 possessions. |
| DEF_RATING | Defensive Rating | Points allowed by the lineup per 100 possessions. Lower is better. |
| NET_RATING | Net Rating | The point differential per 100 possessions (OFF_RATING - DEF_RATING). Positive means the lineup outscores opponents. |
| E_OFF_RATING | Estimated Offensive Rating | An estimation of offensive efficiency using a simplified formula for possessions. |
| E_DEF_RATING | Estimated Defensive Rating | An estimation of defensive efficiency. |
| E_NET_RATING | Estimated Net Rating | The estimated point differential per 100 possessions. |

---

## Possession & Playstyle

| Variable | Full Name | Description |
|---|---|---|
| PACE | Pace | The average number of possessions the lineup plays per 48 minutes. High = fast tempo. |
| E_PACE | Estimated Pace | An estimated calculation of the lineup's pace. |
| PACE_PER40 | Pace per 40 Minutes | The number of possessions scaled to a 40-minute game. |
| POSS | Total Possessions | The actual count of offensive possessions played by this group. |
| AST_PCT | Assist Percentage | Percentage of the lineup's made field goals that were assisted. |
| AST_TO | Assist-to-Turnover Ratio | Number of assists generated for every turnover committed. Higher is better. |
| AST_RATIO | Assist Ratio | The percentage of a lineup's possessions that end in an assist. |
| TM_TOV_PCT | Team Turnover Percentage | Percentage of possessions that end in a turnover. Lower is better. |

---

## Shooting & Rebounding

| Variable | Full Name | Description |
|---|---|---|
| EFG_PCT | Effective Field Goal Percentage | Adjusts field goal percentage to account for a 3-pointer being worth 1.5x a 2-pointer. |
| TS_PCT | True Shooting Percentage | The most complete shooting efficiency metric. Accounts for 2-pointers, 3-pointers, and free throws. |
| OREB_PCT | Offensive Rebound Percentage | Percentage of available offensive rebounds the lineup secures while on the floor. |
| DREB_PCT | Defensive Rebound Percentage | Percentage of available defensive rebounds the lineup secures while on the floor. |
| REB_PCT | Total Rebound Percentage | The percentage of all available rebounds (both ends) grabbed by the lineup. |

---

## Impact

| Variable | Full Name | Description |
|---|---|---|
| PIE | Player Impact Estimate | A comprehensive metric measuring a lineup's overall statistical contribution to the game's positive events. |

---

## Rank Variables

Every base metric has a corresponding `_RANK` column. Rank variables indicate where the lineup stands relative to all other lineups in the dataset.

- **Rank 1** = Best performing lineup in that metric
- **High Rank (e.g., 300)** = Worst performing lineup in that metric

### Key Rank Indicators

| Variable | What a high rank number (poor ranking) means |
|---|---|
| DEF_RATING_RANK | Lineup bleeds points defensively |
| OFF_RATING_RANK | Lineup struggles to score efficiently |
| NET_RATING_RANK | Lineup is a net-negative on the floor and gets outscored |
| TS_PCT_RANK | Lineup shoots poorly across the board |
| EFG_PCT_RANK | Lineup takes inefficient shots or misses frequently |
| REB_PCT_RANK | Lineup is severely out-rebounded |
| TM_TOV_PCT_RANK | Lineup turns the ball over at a high rate |
| PIE_RANK | Lineup has a minimal or negative impact on the game |

> **Note:** The dataset also includes ranks for volume metrics like `GP_RANK`, `W_RANK`, `L_RANK`, `W_PCT_RANK`, `MIN_RANK`, `AST_PCT_RANK`, `AST_TO_RANK`, `AST_RATIO_RANK`, `OREB_PCT_RANK`, `DREB_PCT_RANK`, `PACE_RANK`.
