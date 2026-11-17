import numpy as np
import pandas as pd

from src.data_loader import FEATURES, F1_POINTS


def run_championship_simulation(
    pre_race_df: pd.DataFrame,
    model,
    n_sim: int = 10_000,
    noise_std: float = 2.5,
    simulate_dnfs: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo simulations for the target race and compute WDC title probabilities
    plus full finishing-position distributions.
    """
    drivers = pre_race_df["Abbreviation"].tolist()
    teams = pre_race_df["TeamName"].tolist()
    current_pts = pre_race_df["current_points"].to_numpy(dtype=float)
    current_wins = (
        pre_race_df["current_wins"].to_numpy(dtype=int)
        if "current_wins" in pre_race_df.columns
        else np.zeros(len(drivers), dtype=int)
    )
    dnf_rates = pre_race_df["dnf_rate"].to_numpy(dtype=float)
    model_features = list(getattr(model, "feature_names_in_", FEATURES))
    base_scores = model.predict(pre_race_df[model_features])

    rng = np.random.default_rng(seed)
    n_drv = len(drivers)

    champ_counts = np.zeros(n_drv, dtype=int)
    podium_counts = np.zeros(n_drv, dtype=int)
    win_counts = np.zeros(n_drv, dtype=int)
    dnf_counts = np.zeros(n_drv, dtype=int)
    pos_matrix = np.zeros((n_sim, n_drv), dtype=int)

    points_lookup = np.array([F1_POINTS.get(p, 0) for p in range(1, n_drv + 1)], dtype=float)

    for i in range(n_sim):
        noise = rng.normal(0.0, noise_std, n_drv)
        scores = base_scores + noise

        if simulate_dnfs:
            is_dnf = rng.random(n_drv) < dnf_rates
            dnf_counts += is_dnf.astype(int)
            # Push DNF drivers to the back of the finishing order
            scores = np.where(is_dnf, scores + 50.0, scores)

        order = np.argsort(scores)
        pos = np.empty(n_drv, dtype=int)
        pos[order] = np.arange(1, n_drv + 1)

        pos_matrix[i] = pos
        win_counts += (pos == 1).astype(int)
        podium_counts += (pos <= 3).astype(int)

        race_pts = points_lookup[pos - 1]
        total_pts = current_pts + race_pts
        total_wins = current_wins + (pos == 1).astype(int)

        # FIA tie-breaker: total points, then total wins
        tie_break_score = total_pts + total_wins * 0.01
        champ_counts[np.argmax(tie_break_score)] += 1

    leader_pts = float(np.max(current_pts))
    gaps = leader_pts - current_pts

    sim_df = pd.DataFrame(
        {
            "Driver": drivers,
            "Team": teams,
            "Grid": pre_race_df["grid_pos"].round(1).tolist(),
            "Current Points": current_pts.astype(int),
            "Gap to Leader": gaps.astype(int),
            "Expected Finish": np.round(pos_matrix.mean(axis=0), 2),
            "Race Win %": np.round(win_counts / n_sim * 100, 2),
            "Podium %": np.round(podium_counts / n_sim * 100, 2),
            "DNF %": np.round(dnf_counts / n_sim * 100, 1),
            "Title Prob %": np.round(champ_counts / n_sim * 100, 2),
        }
    ).sort_values(["Title Prob %", "Current Points"], ascending=[False, False]).reset_index(drop=True)

    pos_df = pd.DataFrame(pos_matrix, columns=drivers)
    return sim_df, pos_df


def build_title_scenarios_summary(pre_race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a clear mathematical breakdown for every driver still mathematically eligible
    for the World Championship entering the finale (gap <= 25 points).
    """
    df_sorted = pre_race_df.sort_values(
        ["current_points", "current_wins"], ascending=[False, False]
    ).reset_index(drop=True)
    leader = df_sorted.iloc[0]
    leader_pts = int(leader["current_points"])
    leader_drv = leader["Abbreviation"]

    contenders = df_sorted[leader_pts - df_sorted["current_points"] <= 25].copy()
    rows = []
    for idx, row in contenders.iterrows():
        pts = int(row["current_points"])
        gap = leader_pts - pts
        drv = row["Abbreviation"]
        if idx == 0:
            # Check gap to P2
            p2_pts = int(df_sorted.iloc[1]["current_points"]) if len(df_sorted) > 1 else pts
            lead_margin = pts - p2_pts
            needed_pts = max(0, 25 - lead_margin + 1)
            if needed_pts == 0:
                cond = "Already Champion"
            elif needed_pts <= 15:
                cond = f"Clinches with P podium / ≥ {needed_pts} pts (even if P2 wins)"
            else:
                cond = f"Needs ≥ {needed_pts} pts if P2 wins, or finish ahead of rivals"
        else:
            min_pts_needed = gap + 1
            if min_pts_needed > 25:
                cond = "Needs win (25 pts) + tie-break & leader scores 0"
            elif min_pts_needed > 18:
                cond = f"Must WIN (P1, 25 pts) and {leader_drv} scores < {25 - gap} pts"
            elif min_pts_needed > 15:
                cond = f"Needs P1–P2 (≥ {min_pts_needed} pts) + {leader_drv} finishes low"
            elif min_pts_needed > 12:
                cond = f"Needs Podium (P1–P3, ≥ {min_pts_needed} pts) + outscore {leader_drv} by {min_pts_needed} pts"
            else:
                cond = f"Outscore {leader_drv} by ≥ {min_pts_needed} pts"

        rows.append(
            {
                "Rank": f"P{idx + 1}",
                "Driver": drv,
                "Team": row["TeamName"],
                "Points": pts,
                "Wins": int(row["current_wins"]),
                "Gap": "LEADER" if gap == 0 else f"-{gap} pts",
                "Max Possible Pts": pts + 25,
                "Mathematical Title Condition": cond,
            }
        )
    return pd.DataFrame(rows)
