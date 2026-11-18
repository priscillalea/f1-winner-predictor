import json
import os
import warnings
import pandas as pd
import fastf1
import streamlit as st

warnings.filterwarnings("ignore")

YEAR = 2025
DEFAULT_LAST_ROUND = 23  # Qatar GP — last known race before Abu Dhabi
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

TEAM_COLORS = {
    "McLaren": "#FF8000",
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0090FF",
    "Haas F1 Team": "#B6BABD",
    "Racing Bulls": "#6692FF",
    "Williams": "#64C4FF",
    "Kick Sauber": "#52E252",
}

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
FEATURES = ["grid_pos", "driver_avg_pos", "rolling_3_pos", "team_avg_pos", "dnf_rate"]


@st.cache_data(show_spinner="Loading 2025 F1 season data...")
def load_season(year: int = YEAR, up_to_round: int = DEFAULT_LAST_ROUND) -> pd.DataFrame:
    """Load race results for all completed rounds up to and including `up_to_round`."""
    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_path = os.path.join(DATA_DIR, f"season_{year}_r1_{up_to_round}.csv")
    if os.path.exists(snapshot_path):
        df_all = pd.read_csv(snapshot_path, parse_dates=["EventDate"])
        df_all["GridPosition"] = pd.to_numeric(df_all["GridPosition"], errors="coerce").fillna(20)
        df_all["Position"] = pd.to_numeric(df_all["Position"], errors="coerce").fillna(20)
        df_all["Points"] = pd.to_numeric(df_all["Points"], errors="coerce").fillna(0)
        df_all["DNF"] = pd.to_numeric(df_all["DNF"], errors="coerce").fillna(0).astype(int)
        df_all["Round"] = pd.to_numeric(df_all["Round"], errors="coerce").astype(int)
        return df_all

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["EventFormat"] != "testing"].reset_index(drop=True)

    records = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        if rnd > up_to_round:
            continue
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(telemetry=False, laps=False, weather=False)
            df = session.results[
                [
                    "Abbreviation",
                    "FullName",
                    "TeamName",
                    "TeamColor",
                    "GridPosition",
                    "Position",
                    "Points",
                    "Status",
                ]
            ].copy()
            df["Round"] = rnd
            df["EventName"] = event["EventName"]
            df["EventDate"] = pd.to_datetime(event["EventDate"])
            records.append(df)
        except Exception:
            continue

    df_all = pd.concat([pd.DataFrame(r.to_dict(orient="list")) for r in records], ignore_index=True)
    df_all["GridPosition"] = pd.to_numeric(df_all["GridPosition"], errors="coerce").fillna(20)
    df_all["Position"] = pd.to_numeric(df_all["Position"], errors="coerce").fillna(20)
    df_all["Points"] = pd.to_numeric(df_all["Points"], errors="coerce").fillna(0)
    df_all["DNF"] = (~df_all["Status"].astype(str).str.contains("Finished", na=False)).astype(int)
    df_all.to_csv(snapshot_path, index=False)
    return df_all


@st.cache_data(show_spinner=False)
def load_official_grid(year: int = YEAR, target_round: int = 24) -> dict[str, float]:
    """Load official starting grid positions for `target_round` (pre-race info)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_path = os.path.join(DATA_DIR, f"official_grid_{year}_r{target_round}.json")
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    try:
        session = fastf1.get_session(year, target_round, "R")
        session.load(telemetry=False, laps=False, weather=False)
        res = pd.DataFrame(session.results[["Abbreviation", "GridPosition"]].to_dict(orient="list"))
        res["GridPosition"] = pd.to_numeric(res["GridPosition"], errors="coerce").fillna(20)
        res.loc[res["GridPosition"] <= 0, "GridPosition"] = 20
        grid = {str(k): float(v) for k, v in zip(res["Abbreviation"], res["GridPosition"])}
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2)
        return grid
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def build_features(_df: pd.DataFrame, feature_names: tuple = tuple(FEATURES)) -> pd.DataFrame:
    """Build temporal rolling features without data leakage, including 3-race recent form."""
    df_sorted = pd.DataFrame(_df.to_dict(orient="list")).sort_values(["Round", "Position"]).reset_index(drop=True)
    rows = []
    for rnd in sorted(df_sorted["Round"].unique()):
        past = df_sorted[df_sorted["Round"] < rnd]
        curr = df_sorted[df_sorted["Round"] == rnd].copy()
        if past.empty:
            curr["driver_avg_pos"] = 10.0
            curr["rolling_3_pos"] = 10.0
            curr["team_avg_pos"] = 10.0
            curr["dnf_rate"] = 0.0
        else:
            drv_mean = past.groupby("Abbreviation")["Position"].mean()
            recent_rounds = sorted(past["Round"].unique())[-3:]
            drv_roll3 = past[past["Round"].isin(recent_rounds)].groupby("Abbreviation")["Position"].mean()
            team_mean = past.groupby("TeamName")["Position"].mean()
            drv_dnf = past.groupby("Abbreviation")["DNF"].mean()

            curr["driver_avg_pos"] = curr["Abbreviation"].map(drv_mean).fillna(10.0)
            curr["rolling_3_pos"] = curr["Abbreviation"].map(drv_roll3).fillna(curr["driver_avg_pos"])
            curr["team_avg_pos"] = curr["TeamName"].map(team_mean).fillna(10.0)
            curr["dnf_rate"] = curr["Abbreviation"].map(drv_dnf).fillna(0.0)
        rows.append(curr)

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"GridPosition": "grid_pos"})
    return out


def get_pre_race_features(
    df: pd.DataFrame,
    target_round: int,
    grid_scenario: str = "Official Qualifying Grid",
    custom_grid: dict | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix for `target_round` using ONLY data strictly before `target_round`.
    Avoids leaking the previous race's grid position into the upcoming race.
    """
    past = df[df["Round"] < target_round].copy()
    if past.empty:
        past = df.copy()

    latest_rnd = past["Round"].max()
    drivers_df = (
        past[past["Round"] == latest_rnd][["Abbreviation", "FullName", "TeamName"]]
        .drop_duplicates(subset=["Abbreviation"])
        .reset_index(drop=True)
    )

    drv_mean = past.groupby("Abbreviation")["Position"].mean()
    recent_rounds = sorted(past["Round"].unique())[-3:]
    drv_roll3 = past[past["Round"].isin(recent_rounds)].groupby("Abbreviation")["Position"].mean()
    team_mean = past.groupby("TeamName")["Position"].mean()
    drv_dnf = past.groupby("Abbreviation")["DNF"].mean()
    drv_avg_grid = past.groupby("Abbreviation")["GridPosition"].mean()
    wdc_pts = past.groupby("Abbreviation")["Points"].sum()
    wdc_wins = past[past["Position"] == 1].groupby("Abbreviation")["Position"].count()

    drivers_df["driver_avg_pos"] = drivers_df["Abbreviation"].map(drv_mean).fillna(10.0)
    drivers_df["rolling_3_pos"] = drivers_df["Abbreviation"].map(drv_roll3).fillna(drivers_df["driver_avg_pos"])
    drivers_df["team_avg_pos"] = drivers_df["TeamName"].map(team_mean).fillna(10.0)
    drivers_df["dnf_rate"] = drivers_df["Abbreviation"].map(drv_dnf).fillna(0.0)
    drivers_df["current_points"] = drivers_df["Abbreviation"].map(wdc_pts).fillna(0.0)
    drivers_df["current_wins"] = drivers_df["Abbreviation"].map(wdc_wins).fillna(0).astype(int)

    if grid_scenario == "Official Qualifying Grid":
        official_grid = load_official_grid(YEAR, target_round)
        if official_grid:
            drivers_df["grid_pos"] = drivers_df["Abbreviation"].map(official_grid).fillna(
                drivers_df["Abbreviation"].map(drv_avg_grid).fillna(10.0)
            )
        else:
            drivers_df["grid_pos"] = drivers_df["Abbreviation"].map(drv_avg_grid).fillna(10.0).round(1)
    elif grid_scenario == "WDC Standings Order":
        drivers_df = drivers_df.sort_values("current_points", ascending=False).reset_index(drop=True)
        drivers_df["grid_pos"] = range(1, len(drivers_df) + 1)
    elif grid_scenario == "Custom Grid" and custom_grid:
        drivers_df["grid_pos"] = drivers_df["Abbreviation"].map(custom_grid).fillna(
            drivers_df["Abbreviation"].map(drv_avg_grid).fillna(10.0).round(1)
        )
    else:  # "Season Avg Grid"
        drivers_df["grid_pos"] = drivers_df["Abbreviation"].map(drv_avg_grid).fillna(10.0).round(1)

    return drivers_df
