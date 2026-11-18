import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import (
    YEAR,
    DEFAULT_LAST_ROUND,
    TEAM_COLORS,
    FEATURES,
    load_season,
    build_features,
    get_pre_race_features,
)
from src.ml_models import train_and_evaluate
from src.simulator import run_championship_simulation, build_title_scenarios_summary

# ---------------------------------------------------------------------------
# Page Config & F1 Broadcast Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Analytics · 2025 Season & Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0D10;
    color: #F3F4F6;
}

h1, h2, h3, .f1-title {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Top Hero Banner */
.f1-hero {
    background: linear-gradient(135deg, #14171D 0%, #0B0D10 100%);
    border-left: 5px solid #E10600;
    border-top: 1px solid #232832;
    border-right: 1px solid #232832;
    border-bottom: 1px solid #232832;
    border-radius: 6px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.f1-hero-kicker {
    color: #E10600;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.f1-hero-title {
    color: #FFFFFF;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0;
}
.f1-hero-sub {
    color: #9CA3AF;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

/* KPI Cards */
.kpi-card {
    background-color: #14171D;
    border: 1px solid #232832;
    border-top: 3px solid #E10600;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    height: 100%;
}
.kpi-label {
    color: #9CA3AF;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.kpi-value {
    color: #FFFFFF;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
}
.kpi-sub {
    color: #9CA3AF;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

/* Broadcast Driver Podium Cards */
.driver-card {
    background: linear-gradient(160deg, #161A22 0%, #0E1015 100%);
    border: 1px solid #252A35;
    border-radius: 8px;
    padding: 1.15rem 1.35rem;
    position: relative;
    overflow: hidden;
}
.driver-card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}
.driver-pos-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1;
}
.driver-gap-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    background-color: #1F2430;
    color: #E5E7EB;
    letter-spacing: 0.05em;
}
.driver-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #FFFFFF;
    letter-spacing: 0.03em;
    margin: 0.15rem 0;
}
.driver-team {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.85rem;
}
.driver-stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    border-top: 1px solid #232832;
    padding-top: 0.7rem;
}
.driver-stat-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
}
.driver-stat-lbl {
    font-size: 0.68rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Section Divider */
.f1-section {
    border-bottom: 1px solid #232832;
    padding-bottom: 0.4rem;
    margin: 1.6rem 0 1rem 0;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #FFFFFF;
}
.f1-accent {
    color: #E10600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0F1217;
    border-right: 1px solid #232832;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Premium refinement layer (overrides base styles; purely visual)
REFINED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --f1-red: #E10600; --f1-red-dim: rgba(225,6,0,0.14);
    --bg: #08090B; --panel: #111318; --panel-2: #16191F;
    --line: #22262E; --muted: #8B93A1; --text: #F5F6F8;
}
.stApp { background:
    radial-gradient(1200px 400px at 85% -10%, rgba(225,6,0,0.08), transparent 60%),
    linear-gradient(180deg, #0A0B0E 0%, #07080A 100%); }
.block-container { padding-top: 1.6rem !important; padding-bottom: 2rem !important; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, .f1-title, .f1-hero-title, .kpi-value, .driver-name, .f1-section {
    font-family: 'Titillium Web', 'Barlow Condensed', sans-serif !important;
}

/* Hero: compact header strip */
.f1-hero { padding: 1rem 1.4rem; margin-bottom: 1rem; border-radius: 4px;
    background: linear-gradient(90deg, #12141A 0%, #0B0C10 70%);
    border: 1px solid var(--line); border-left: 4px solid var(--f1-red); }
.f1-hero-title { font-size: 2rem; font-weight: 900; letter-spacing: 0.02em; }
.f1-hero-sub { font-size: 0.88rem; margin-top: 0.3rem; color: var(--muted); }
.f1-hero-kicker { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.18em; }

/* KPI cards */
.kpi-card { background: var(--panel); border: 1px solid var(--line); border-top: none;
    border-radius: 4px; padding: 0.85rem 1.1rem; position: relative;
    transition: border-color .2s ease, transform .2s ease; }
.kpi-card::before { content: ""; position: absolute; left: 0; top: 0; height: 2px; width: 36px; background: var(--f1-red); }
.kpi-card:hover { border-color: #3A3F4A; transform: translateY(-1px); }
.kpi-label { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.14em; color: var(--muted); }
.kpi-value { font-size: 1.75rem; font-weight: 700; }
.kpi-sub { font-size: 0.75rem; }

/* Winner prediction hero */
.win-hero { position: relative; overflow: hidden; border-radius: 6px; padding: 1.3rem 1.6rem;
    background: linear-gradient(115deg, #15171D 0%, #0D0E12 55%, #0A0B0E 100%);
    border: 1px solid var(--line); height: 252px; box-sizing: border-box;
    display: flex; flex-direction: column; justify-content: space-between; }
.win-hero::after { content: ""; position: absolute; inset: 0 0 0 45%;
    background: repeating-linear-gradient(135deg, rgba(255,255,255,0.03) 0 2px, transparent 2px 14px);
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 60%);
    mask-image: linear-gradient(90deg, transparent 0%, #000 60%); pointer-events: none; }
.win-tag { display: inline-flex; align-items: center; gap: .5rem; font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; letter-spacing: .18em; color: var(--f1-red); text-transform: uppercase; }
.win-tag .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--f1-red);
    box-shadow: 0 0 0 0 rgba(225,6,0,.6); animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(225,6,0,.55);} 70% { box-shadow: 0 0 0 10px rgba(225,6,0,0);} 100% { box-shadow: 0 0 0 0 rgba(225,6,0,0);} }
.win-race { color: var(--muted); font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; margin-top: .35rem; }
.win-name { font-family: 'Titillium Web', sans-serif; font-weight: 900; font-size: 3rem; line-height: 1;
    text-transform: uppercase; color: #fff; margin: .55rem 0 .15rem 0; letter-spacing: .01em; }
.win-team { font-weight: 700; font-size: .85rem; letter-spacing: .12em; text-transform: uppercase; }
.win-prob { font-family: 'Titillium Web', sans-serif; font-weight: 900; font-size: 3.4rem; line-height: 1; color: #fff; }
.win-prob small { font-size: 1.3rem; color: var(--muted); font-weight: 600; }
.win-stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; padding-top: .8rem; border-top: 1px solid var(--line); position: relative; z-index: 1; }
.win-stat-v { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; color: #fff; }
.win-stat-l { font-size: .64rem; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }

/* Timing-tower ranking */
.tower { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: .9rem 1rem;
    height: 252px; box-sizing: border-box; display: flex; flex-direction: column; }
.tower-h { font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .16em; color: var(--muted);
    text-transform: uppercase; display: flex; justify-content: space-between; margin-bottom: .3rem; }
.tower-row { display: grid; grid-template-columns: 28px 4px 1fr 64px; align-items: center; gap: .55rem;
    flex: 1; padding: 0 .3rem; border-bottom: 1px solid #1B1E25; transition: background .15s ease; }
.tower-row:last-child { border-bottom: none; }
.tower-row:hover { background: #181B22; }
.tower-pos { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff; font-size: .85rem; text-align: center; }
.tower-bar { height: 18px; border-radius: 1px; }
.tower-drv { font-family: 'Titillium Web', sans-serif; font-weight: 700; color: #fff; letter-spacing: .04em; }
.tower-drv span { color: var(--muted); font-weight: 400; font-size: .75rem; margin-left: .4rem; }
.tower-val { font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: #fff; text-align: right; }
.tower-row.lead .tower-pos { color: var(--f1-red); }

/* Driver cards */
.driver-card { border-radius: 4px; padding: .95rem 1.15rem; transition: transform .2s ease, border-color .2s ease; }
.driver-card:hover { transform: translateY(-2px); border-color: #3A3F4A; }
.driver-name { font-size: 1.3rem; }
.driver-pos-badge { font-family: 'JetBrains Mono', monospace !important; font-size: 1.3rem; }
.driver-gap-badge { font-family: 'JetBrains Mono', monospace !important; font-size: .68rem; }

/* Section headers */
.f1-section { font-size: 1.05rem; letter-spacing: .1em; margin: 1.3rem 0 .8rem 0; padding-bottom: .45rem;
    display: flex; align-items: center; gap: .6rem; }
.f1-accent { font-family: 'JetBrains Mono', monospace; font-size: .8rem; }

.note { font-size: .74rem; color: var(--muted); margin-top: .45rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0A0B0E; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: .55rem .7rem; border-radius: 3px; border-left: 2px solid transparent;
    transition: background .15s ease, border-color .15s ease; margin: 0; width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #14161C; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: var(--f1-red-dim); border-left-color: var(--f1-red); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] p {
    font-family: 'Titillium Web', sans-serif; font-weight: 600; font-size: .92rem; letter-spacing: .03em; }
.side-meta { font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--muted); line-height: 1.8; }
.side-meta b { color: #fff; font-weight: 700; }

/* Widgets */
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
div[data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: .3rem; }
.stAlert { border-radius: 4px; }
</style>
"""
st.markdown(REFINED_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#C9CED6", family="Titillium Web, Inter, sans-serif", size=12),
    title_font=dict(family="Titillium Web, sans-serif", size=14, color="#FFFFFF"),
    margin=dict(l=40, r=20, t=48, b=40),
    xaxis=dict(gridcolor="#1C1F26", zerolinecolor="#2A2E37", linecolor="#2A2E37"),
    yaxis=dict(gridcolor="#1C1F26", zerolinecolor="#2A2E37", linecolor="#2A2E37"),
    hoverlabel=dict(bgcolor="#16191F", bordercolor="#E10600", font=dict(family="JetBrains Mono, monospace", color="#FFFFFF")),
)


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def driver_broadcast_card(rank: int, row: pd.Series, leader_pts: float):
    color = row["Color"]
    gap = leader_pts - row["Points"]
    gap_str = "Championship Leader" if gap == 0 else f"-{gap:.0f} PTS TO LEADER"
    st.markdown(
        f"""
        <div class="driver-card" style="border-left: 5px solid {color};">
            <div class="driver-card-header">
                <span class="driver-pos-badge">P{rank}</span>
                <span class="driver-gap-badge" style="color: {'#E10600' if gap == 0 else '#D1D5DB'};">{gap_str.upper()}</span>
            </div>
            <div class="driver-name">{row['FullName']} <span style="color:{color};">({row['Abbreviation']})</span></div>
            <div class="driver-team" style="color: {color};">{row['TeamName']}</div>
            <div class="driver-stats-grid">
                <div>
                    <div class="driver-stat-val">{row['Points']:.0f}</div>
                    <div class="driver-stat-lbl">Points</div>
                </div>
                <div>
                    <div class="driver-stat-val">{row['Wins']}</div>
                    <div class="driver-stat-lbl">Wins</div>
                </div>
                <div>
                    <div class="driver-stat-val">{row['Podiums']}</div>
                    <div class="driver-stat-lbl">Podiums</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load Data & Train Models (Cached)
# ---------------------------------------------------------------------------
df = load_season(YEAR, DEFAULT_LAST_ROUND)
df_feat = build_features(df, tuple(FEATURES))
if not all(f in df_feat.columns for f in FEATURES):
    st.cache_data.clear()
    st.cache_resource.clear()
    df = load_season(YEAR, DEFAULT_LAST_ROUND)
    df_feat = build_features(df, tuple(FEATURES))

ml_res = train_and_evaluate(df_feat, tuple(FEATURES))
if list(getattr(ml_res["best_model"], "feature_names_in_", [])) != list(FEATURES):
    st.cache_resource.clear()
    ml_res = train_and_evaluate(df_feat, tuple(FEATURES))

wdc = (
    df.groupby(["Abbreviation", "FullName", "TeamName"], as_index=False)
    .agg(
        Points=("Points", "sum"),
        Wins=("Position", lambda s: int((s == 1).sum())),
        Podiums=("Position", lambda s: int((s <= 3).sum())),
        AvgFinish=("Position", "mean"),
    )
    .sort_values(["Points", "Wins"], ascending=[False, False])
    .reset_index(drop=True)
)
wdc["Rank"] = range(1, len(wdc) + 1)
wdc["Color"] = wdc["TeamName"].map(TEAM_COLORS).fillna("#888888")

wcc = (
    df.groupby("TeamName", as_index=False)["Points"]
    .sum()
    .sort_values("Points", ascending=False)
    .reset_index(drop=True)
)
wcc["Color"] = wcc["TeamName"].map(TEAM_COLORS).fillna("#888888")

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
_best_mae_val = ml_res["eval_df"].loc[ml_res["eval_df"]["Model"] == ml_res["best_name"], "Test MAE"].values[0]

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.2rem 0 1rem 0; border-bottom: 1px solid #22262E; margin-bottom: 0.9rem;">
            <div style="display:flex; align-items:center; gap:.55rem;">
                <div style="width:6px; height:30px; background:#E10600; transform:skewX(-14deg);"></div>
                <div>
                    <div style="color:#FFFFFF; font-family:'Titillium Web',sans-serif; font-weight:900; font-size:1.35rem; letter-spacing:0.06em; line-height:1;">
                        F1 <span style="color:#E10600;">WINNER</span> PREDICTOR
                    </div>
                    <div style="color:#8B93A1; font-family:'JetBrains Mono',monospace; font-size:0.64rem; letter-spacing:0.16em; text-transform:uppercase; margin-top:.25rem;">
                        2025 Season · Data &amp; ML
                    </div>
                </div>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:.62rem; letter-spacing:.18em; color:#8B93A1; margin:.2rem 0 .4rem .2rem;">MODULES</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "01 — Season Overview",
            "02 — Race & Driver Analysis",
            "03 — Machine Learning Lab",
            "04 — Championship Simulator",
            "05 — About the Model",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div style="border-top:1px solid #22262E; margin-top:1rem; padding-top:.9rem;" class="side-meta">
            <div>DATASET &nbsp;<b>{df['Round'].nunique()} rounds · {len(df)} entries</b></div>
            <div>BEST MODEL &nbsp;<b>{ml_res['best_name']}</b></div>
            <div>TEST MAE &nbsp;<b>{_best_mae_val:.2f} pos</b></div>
            <div>SOURCE &nbsp;<b>FastF1</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Running 10,000 race simulations for the Abu Dhabi forecast...")
def _default_forecast(model_name: str, feature_key: tuple):
    """Baseline forecast = simulator defaults (official grid, σ 2.5, DNFs on, 10k runs) with the best model."""
    pre = get_pre_race_features(df, target_round=24, grid_scenario="Official Qualifying Grid")
    sim, _ = run_championship_simulation(
        pre_race_df=pre, model=ml_res["models"][model_name], n_sim=10_000, noise_std=2.5, simulate_dnfs=True
    )
    return sim


def winner_hero(sim: pd.DataFrame, model_name: str):
    top = sim.sort_values("Race Win %", ascending=False).iloc[0]
    full = wdc.set_index("Abbreviation")["FullName"].to_dict().get(top["Driver"], top["Driver"])
    color = TEAM_COLORS.get(top["Team"], "#888888")
    st.markdown(
        f"""
        <div class="win-hero" style="border-left:4px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; position:relative; z-index:1;">
                <div>
                    <div class="win-tag"><span class="dot"></span>Predicted Race Winner</div>
                    <div class="win-race">Round 24 · Abu Dhabi Grand Prix · Season Finale</div>
                    <div class="win-name">{full}</div>
                    <div class="win-team" style="color:{color};">{top['Team']} · {top['Driver']}</div>
                </div>
                <div style="text-align:right;">
                    <div class="win-stat-l">Win probability</div>
                    <div class="win-prob">{top['Race Win %']:.1f}<small>%</small></div>
                </div>
            </div>
            <div class="win-stats">
                <div><div class="win-stat-v">P{top['Grid']:.0f}</div><div class="win-stat-l">Grid</div></div>
                <div><div class="win-stat-v">P{top['Expected Finish']:.2f}</div><div class="win-stat-l">Expected finish</div></div>
                <div><div class="win-stat-v">{top['Podium %']:.1f}%</div><div class="win-stat-l">Podium</div></div>
                <div><div class="win-stat-v">{top['Title Prob %']:.1f}%</div><div class="win-stat-l">Title</div></div>
                <div><div class="win-stat-v">{model_name}</div><div class="win-stat-l">Model</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def win_tower(sim: pd.DataFrame, n: int = 6):
    rows = sim.sort_values("Race Win %", ascending=False).head(n)
    html = '<div class="tower"><div class="tower-h"><span>Win Probability · Top 6</span><span>Monte Carlo</span></div>'
    for i, (_, r) in enumerate(rows.iterrows()):
        c = TEAM_COLORS.get(r["Team"], "#888888")
        html += (
            f'<div class="tower-row {"lead" if i == 0 else ""}"><div class="tower-pos">{i + 1}</div>'
            f'<div class="tower-bar" style="background:{c};"></div>'
            f'<div class="tower-drv">{r["Driver"]}<span>{r["Team"]}</span></div>'
            f'<div class="tower-val">{r["Race Win %"]:.1f}%</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ===========================================================================
# PAGE 01 — SEASON OVERVIEW
# ===========================================================================
if page == "01 — Season Overview":
    st.markdown(
        """
        <div class="f1-hero" style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="f1-hero-kicker">2025 SEASON · DATA → MACHINE LEARNING → PREDICTION</div>
                <h1 class="f1-hero-title">F1 WINNER PREDICTOR</h1>
                <div class="f1-hero-sub">Who wins the 2025 season finale? A model trained on 23 rounds of official race data, stress-tested with 10,000 Monte Carlo simulations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 01 // Headline prediction (first thing on screen)
    forecast = _default_forecast("Random Forest", tuple(FEATURES))
    col_win, col_tower = st.columns([1.75, 1])
    with col_win:
        winner_hero(forecast, "Random Forest")
    with col_tower:
        win_tower(forecast)
    st.markdown(
        '<div class="note">Baseline scenario: official qualifying grid · σ = 2.5 positions · stochastic DNFs on · 10,000 runs. '
        'Probabilities depend on these assumptions — adjust them in <b>04 — Championship Simulator</b>.</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    best_row = ml_res["eval_df"][ml_res["eval_df"]["Model"] == ml_res["best_name"]].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("RACE WEEKENDS", str(df["Round"].nunique()), "Rounds 1–23 (up to Qatar)")
    with c2:
        kpi_card("DRIVER RECORDS", str(len(df)), "Official FastF1 race entries")
    with c3:
        kpi_card("BEST MODEL", ml_res["best_name"], f"Top-3 Precision: {best_row['Top-3 Precision']:.1%}")
    with c4:
        kpi_card("TEST MAE", f"{best_row['Test MAE']:.2f}", "Positions (temporal split)")

    # Top 3 Championship Contenders Broadcast Cards
    st.markdown('<div class="f1-section"><span class="f1-accent">02 //</span> TITLE CONTENDERS — TOP 3 WDC</div>', unsafe_allow_html=True)
    leader_pts = float(wdc.iloc[0]["Points"])
    p1_col, p2_col, p3_col = st.columns(3)
    for idx, col in enumerate([p1_col, p2_col, p3_col]):
        with col:
            driver_broadcast_card(idx + 1, wdc.iloc[idx], leader_pts)

    st.markdown('<div class="f1-section"><span class="f1-accent">03 //</span> 2025 CHAMPIONSHIP STANDINGS</div>', unsafe_allow_html=True)

    col_wdc, col_wcc = st.columns([1.25, 1])

    with col_wdc:
        top12 = wdc.head(12).iloc[::-1]
        fig_wdc = go.Figure(
            go.Bar(
                x=top12["Points"],
                y=top12["Abbreviation"],
                orientation="h",
                marker_color=top12["Color"],
                text=top12["Points"].apply(lambda p: f"{p:.0f} pts"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Points: %{x:.0f}<extra></extra>",
            )
        )
        fig_wdc.update_layout(
            **PLOTLY_LAYOUT,
            title="Drivers' Championship (WDC) — Top 12",
            height=430,
            xaxis_title="Points",
            xaxis_range=[0, wdc["Points"].max() * 1.15],
        )
        st.plotly_chart(fig_wdc, width="stretch")

    with col_wcc:
        fig_wcc = go.Figure(
            go.Bar(
                x=wcc["TeamName"],
                y=wcc["Points"],
                marker_color=wcc["Color"],
                text=wcc["Points"].apply(lambda p: f"{p:.0f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Points: %{y:.0f}<extra></extra>",
            )
        )
        fig_wcc.update_layout(
            **PLOTLY_LAYOUT,
            title="Constructors' Championship (WCC)",
            height=430,
            yaxis_title="Points",
            xaxis_tickangle=-35,
            yaxis_range=[0, wcc["Points"].max() * 1.15],
        )
        st.plotly_chart(fig_wcc, width="stretch")

    st.markdown('<div class="f1-section"><span class="f1-accent">04 //</span> CUMULATIVE POINTS EVOLUTION & FULL TABLE</div>', unsafe_allow_html=True)

    top5_drivers = wdc.head(5)["Abbreviation"].tolist()
    df_top5 = df[df["Abbreviation"].isin(top5_drivers)].sort_values("Round").copy()
    df_top5["CumPoints"] = df_top5.groupby("Abbreviation")["Points"].cumsum()

    fig_cum = go.Figure()
    for drv in top5_drivers:
        d = df_top5[df_top5["Abbreviation"] == drv]
        color = TEAM_COLORS.get(d["TeamName"].iloc[0], "#888888")
        fig_cum.add_trace(
            go.Scatter(
                x=d["Round"],
                y=d["CumPoints"],
                mode="lines+markers",
                name=drv,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{drv}</b> · Round %{{x}}<br>Cumulative: %{{y:.0f}} pts<extra></extra>",
            )
        )
    fig_cum.update_layout(
        **PLOTLY_LAYOUT,
        title="Championship Contenders — Points Progression (Rounds 1–23)",
        xaxis_title="Round",
        yaxis_title="Cumulative Points",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cum, width="stretch")

    with st.expander("View Full 2025 Driver Standings Table (Broadcast View)", expanded=False):
        st.dataframe(
            wdc[["Rank", "Abbreviation", "FullName", "TeamName", "Wins", "Podiums", "AvgFinish", "Points"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Pos", format="%d"),
                "Abbreviation": st.column_config.TextColumn("Code"),
                "FullName": st.column_config.TextColumn("Driver"),
                "TeamName": st.column_config.TextColumn("Team"),
                "AvgFinish": st.column_config.NumberColumn("Avg Finish", format="%.2f"),
                "Points": st.column_config.ProgressColumn(
                    "Season Points",
                    format="%.0f pts",
                    min_value=0,
                    max_value=float(wdc["Points"].max()),
                ),
            },
        )


# ===========================================================================
# PAGE 02 — RACE & DRIVER ANALYSIS
# ===========================================================================
elif page == "02 — Race & Driver Analysis":
    st.markdown(
        """
        <div class="f1-hero">
            <div class="f1-hero-kicker">EXPLORATORY DATA ANALYSIS</div>
            <h1 class="f1-hero-title">RACE & DRIVER PERFORMANCE</h1>
            <div class="f1-hero-sub">
                Filter by team, driver, or Grand Prix to inspect grid-to-finish conversion, reliability (DNF), and head-to-head form.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        teams_sel = st.multiselect("Filter by Team", sorted(df["TeamName"].unique()), default=[])
    with f2:
        drivers_sel = st.multiselect("Filter by Driver", sorted(df["Abbreviation"].unique()), default=[])
    with f3:
        events_sel = st.multiselect("Filter by Grand Prix", df.sort_values("Round")["EventName"].unique().tolist(), default=[])

    df_filt = df.copy()
    if teams_sel:
        df_filt = df_filt[df_filt["TeamName"].isin(teams_sel)]
    if drivers_sel:
        df_filt = df_filt[df_filt["Abbreviation"].isin(drivers_sel)]
    if events_sel:
        df_filt = df_filt[df_filt["EventName"].isin(events_sel)]

    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = px.scatter(
            df_filt,
            x="GridPosition",
            y="Position",
            color="TeamName",
            color_discrete_map=TEAM_COLORS,
            hover_data=["Abbreviation", "EventName", "Points"],
            title="Grid Position vs. Finish Position",
            opacity=0.75,
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[1, 20],
                y=[1, 20],
                mode="lines",
                line=dict(color="#6B7280", dash="dash", width=1.5),
                name="No position change",
            )
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Grid Position", yaxis_title="Finish Position")
        st.plotly_chart(fig_scatter, width="stretch")

    with col_b:
        dnf_rate = (
            df_filt.groupby("TeamName", as_index=False)["DNF"]
            .mean()
            .sort_values("DNF", ascending=False)
        )
        dnf_rate["DNF_pct"] = dnf_rate["DNF"] * 100
        dnf_rate["Color"] = dnf_rate["TeamName"].map(TEAM_COLORS).fillna("#888888")

        fig_dnf = go.Figure(
            go.Bar(
                x=dnf_rate["TeamName"],
                y=dnf_rate["DNF_pct"],
                marker_color=dnf_rate["Color"],
                text=dnf_rate["DNF_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            )
        )
        fig_dnf.update_layout(
            **PLOTLY_LAYOUT,
            title="DNF Rate by Team (%)",
            height=420,
            yaxis_title="DNF %",
            xaxis_tickangle=-35,
            yaxis_range=[0, max(dnf_rate["DNF_pct"].max() * 1.2, 5)],
        )
        st.plotly_chart(fig_dnf, width="stretch")

    st.markdown('<div class="f1-section"><span class="f1-accent">02 //</span> HEAD-TO-HEAD DRIVER COMPARISON</div>', unsafe_allow_html=True)

    all_drvs = wdc["Abbreviation"].tolist()
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        drv1 = st.selectbox("Driver A", all_drvs, index=0)
    with c_d2:
        drv2 = st.selectbox("Driver B", all_drvs, index=min(1, len(all_drvs) - 1))

    comp_df = df[df["Abbreviation"].isin([drv1, drv2])].sort_values("Round")
    fig_h2h = go.Figure()
    for d_code in [drv1, drv2]:
        sub = comp_df[comp_df["Abbreviation"] == d_code]
        color = TEAM_COLORS.get(sub["TeamName"].iloc[0], "#888888") if not sub.empty else "#888888"
        fig_h2h.add_trace(
            go.Scatter(
                x=sub["Round"],
                y=sub["Position"],
                mode="lines+markers",
                name=d_code,
                line=dict(color=color, width=2.5),
            )
        )
    fig_h2h.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Race Finish Position per Round: {drv1} vs {drv2} (lower is better)",
        xaxis_title="Round",
        yaxis_title="Finish Position",
        yaxis_autorange="reversed",
        height=350,
    )
    st.plotly_chart(fig_h2h, width="stretch")


# ===========================================================================
# PAGE 03 — MACHINE LEARNING LAB
# ===========================================================================
elif page == "03 — Machine Learning Lab":
    st.markdown(
        """
        <div class="f1-hero">
            <div class="f1-hero-kicker">MODEL TRAINING & EVALUATION</div>
            <h1 class="f1-hero-title">MACHINE LEARNING LAB</h1>
            <div class="f1-hero-sub">
                Temporal train/test split (no data leakage): comparing Random Forest vs. XGBoost with season & 3-race rolling form.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("TRAIN SPLIT", ml_res["train_rounds"], f"{ml_res['train_size']} samples")
    with c2:
        kpi_card("TEST SPLIT", ml_res["test_rounds"], f"{ml_res['test_size']} samples")
    with c3:
        kpi_card("SELECTED MODEL", ml_res["best_name"], "Lowest Test MAE")
    with c4:
        best_mae = ml_res["eval_df"].loc[ml_res["eval_df"]["Model"] == ml_res["best_name"], "Test MAE"].values[0]
        kpi_card("BEST TEST MAE", f"{best_mae:.3f}", "Positions")

    st.markdown('<div class="f1-section"><span class="f1-accent">01 //</span> MODEL COMPARISON & FEATURE IMPORTANCE</div>', unsafe_allow_html=True)

    col_tbl = st.container()
    col_imp = st.container()
    with col_tbl:
        st.dataframe(
            ml_res["eval_df"],
            width="stretch",
            hide_index=True,
            column_config={
                "Train MAE": st.column_config.NumberColumn("Train MAE (pos)", format="%.3f"),
                "Test MAE": st.column_config.NumberColumn("Test MAE (pos)", format="%.3f"),
                "Top-3 Precision": st.column_config.ProgressColumn(
                    "Top-3 Precision",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

    with col_imp:
        best_model = ml_res["best_model"]
        model_features = list(getattr(best_model, "feature_names_in_", FEATURES))
        imp_df = pd.DataFrame(
            {"Feature": model_features, "Importance": best_model.feature_importances_}
        ).sort_values("Importance")

        fig_imp = go.Figure(
            go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color=["#E10600" if i == len(imp_df) - 1 else "#3A3F4A" for i in range(len(imp_df))],
                width=0.5,
                text=imp_df["Importance"].apply(lambda v: f"{v:.2f}"),
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig_imp.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Feature Importance — {ml_res['best_name']}",
            height=260,
        )
        fig_imp.update_xaxes(visible=False, range=[0, imp_df["Importance"].max() * 1.15])
        fig_imp.update_yaxes(showgrid=False)
        st.plotly_chart(fig_imp, width="stretch")

    st.markdown('<div class="f1-section"><span class="f1-accent">02 //</span> RESIDUAL ANALYSIS: WHO OUTPERFORMS THE MODEL?</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model to Inspect", ["Random Forest", "XGBoost"])
    test_df = ml_res["test_df"].copy()
    pred_col = f"pred_{model_choice}"
    # Positive residual = Predicted e.g. P8, Finished P4 -> +4 positions gained vs model expectation
    test_df["Positions_vs_Model"] = test_df[pred_col] - test_df["Position"]

    col_scatter, col_resid = st.columns(2)
    with col_scatter:
        fig_pred = px.scatter(
            test_df,
            x="Position",
            y=pred_col,
            color="TeamName",
            color_discrete_map=TEAM_COLORS,
            hover_data=["Abbreviation", "EventName", "grid_pos"],
            title=f"Actual vs. Predicted Finish — {model_choice}",
            opacity=0.8,
        )
        fig_pred.add_trace(
            go.Scatter(
                x=[1, 20],
                y=[1, 20],
                mode="lines",
                line=dict(color="#E10600", dash="dash", width=1.5),
                name="Perfect Prediction",
            )
        )
        fig_pred.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="Actual Position", yaxis_title="Predicted Position")
        st.plotly_chart(fig_pred, width="stretch")

    with col_resid:
        resid_drv = (
            test_df.groupby(["Abbreviation", "TeamName"], as_index=False)["Positions_vs_Model"]
            .mean()
            .sort_values("Positions_vs_Model", ascending=True)
        )
        resid_drv["Color"] = resid_drv["Positions_vs_Model"].apply(lambda v: "#10B981" if v >= 0 else "#E10600")
        fig_res = go.Figure(
            go.Bar(
                x=resid_drv["Positions_vs_Model"],
                y=resid_drv["Abbreviation"],
                orientation="h",
                marker_color=resid_drv["Color"],
                text=resid_drv["Positions_vs_Model"].apply(lambda v: f"{v:+.2f}"),
                textposition="outside",
            )
        )
        fig_res.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Positions Gained vs. Model Prediction (Green = Beats Model)",
            height=400,
            xaxis_title="Predicted Pos − Actual Pos (Test Set)",
            xaxis_range=[resid_drv["Positions_vs_Model"].min() * 1.25, resid_drv["Positions_vs_Model"].max() * 1.25],
        )
        st.plotly_chart(fig_res, width="stretch")

    st.markdown('<div class="f1-section"><span class="f1-accent">03 //</span> RACE-BY-RACE PODIUM & ORDER INSPECTOR (TEST SET)</div>', unsafe_allow_html=True)

    test_events = (
        test_df[["Round", "EventName"]]
        .drop_duplicates()
        .sort_values("Round")
    )
    event_options = {
        int(r["Round"]): f"Round {int(r['Round'])} — {r['EventName']}"
        for _, r in test_events.iterrows()
    }
    sel_round = st.selectbox("Select Test Grand Prix", list(event_options.keys()), format_func=lambda k: event_options[k])

    race_slice = test_df[test_df["Round"] == sel_round].copy()
    race_slice["Predicted Rank"] = race_slice[pred_col].rank(method="first").astype(int)
    race_slice = race_slice.sort_values("Position").reset_index(drop=True)

    p_actual = race_slice.nsmallest(3, "Position")["Abbreviation"].tolist()
    p_pred = race_slice.nsmallest(3, "Predicted Rank")["Abbreviation"].tolist()

    pc1, pc2 = st.columns(2)
    with pc1:
        kpi_card("ACTUAL PODIUM (P1 · P2 · P3)", " · ".join(p_actual), event_options[sel_round])
    with pc2:
        hits = len(set(p_actual) & set(p_pred))
        kpi_card(f"PREDICTED PODIUM ({model_choice})", " · ".join(p_pred), f"Captured {hits}/3 podium finishers")

    st.write("")
    st.dataframe(
        race_slice[["Position", "Predicted Rank", "Abbreviation", "TeamName", "grid_pos", pred_col, "Positions_vs_Model"]].head(10),
        width="stretch",
        hide_index=True,
        column_config={
            "Position": st.column_config.NumberColumn("Actual Pos", format="P%d"),
            "Predicted Rank": st.column_config.NumberColumn("Pred Rank", format="P%d"),
            "Abbreviation": st.column_config.TextColumn("Driver"),
            "TeamName": st.column_config.TextColumn("Team"),
            "grid_pos": st.column_config.NumberColumn("Grid", format="P%.0f"),
            pred_col: st.column_config.NumberColumn("Raw Pred Score", format="%.2f"),
            "Positions_vs_Model": st.column_config.NumberColumn("Diff vs Model", format="%+.2f"),
        },
    )


# ===========================================================================
# PAGE 04 — CHAMPIONSHIP SIMULATOR
# ===========================================================================
elif page == "04 — Championship Simulator":
    st.markdown(
        """
        <div class="f1-hero">
            <div class="f1-hero-kicker">MONTE CARLO SCENARIO ENGINE</div>
            <h1 class="f1-hero-title">CHAMPIONSHIP SIMULATOR</h1>
            <div class="f1-hero-sub">
                Simulate the 2025 season finale using strictly pre-race historical form, official qualifying or custom grids, and stochastic DNF & pace variance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "**Methodological Note:** All rolling form features (`driver_avg_pos`, `rolling_3_pos`, `team_avg_pos`, `dnf_rate`) "
        "are computed strictly from rounds **prior** to Abu Dhabi. You can use the **Official Qualifying Grid** (pre-race Saturday classification) "
        "or test counterfactual starting grids."
    )

    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        target_round = st.selectbox("Target Race", [24], format_func=lambda r: f"Round {r} — Abu Dhabi GP")
    with col_cfg2:
        grid_scenario = st.selectbox(
            "Starting Grid Scenario",
            ["Official Qualifying Grid", "Season Avg Grid", "WDC Standings Order", "Custom Grid"],
        )
    with col_cfg3:
        sim_model_name = st.selectbox("Prediction Model", ["Random Forest", "XGBoost"])
    with col_cfg4:
        n_sim = st.select_slider("Simulations", options=[1000, 2500, 5000, 10000, 20000], value=10000)

    col_s1, col_s2 = st.columns([2.2, 1])
    with col_s1:
        noise_std = st.slider(
            "Race Pace Unpredictability (Gaussian Noise σ in positions)",
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.25,
            help="Captures safety cars, strategy variance, and lap-1 incidents around the model's base prediction.",
        )
    with col_s2:
        st.write("")
        simulate_dnfs = st.toggle(
            "Simulate Mechanical / Crash DNFs",
            value=True,
            help="Triggers random retirements per driver using their historical 2025 DNF rate.",
        )

    custom_grid = None
    if grid_scenario == "Custom Grid":
        with st.expander("Adjust Custom Starting Grid (Top Contenders)", expanded=True):
            top_contenders = wdc.head(8)["Abbreviation"].tolist()
            custom_grid = {}
            cols = st.columns(4)
            for idx, drv in enumerate(top_contenders):
                with cols[idx % 4]:
                    custom_grid[drv] = st.number_input(f"{drv} Grid", min_value=1, max_value=20, value=idx + 1, key=f"grid_{drv}")

    pre_race_df = get_pre_race_features(df, target_round=target_round, grid_scenario=grid_scenario, custom_grid=custom_grid)
    sim_df, pos_df = run_championship_simulation(
        pre_race_df=pre_race_df,
        model=ml_res["models"][sim_model_name],
        n_sim=n_sim,
        noise_std=noise_std,
        simulate_dnfs=simulate_dnfs,
    )

    # Section 1: Mathematical Title Scenarios
    st.markdown('<div class="f1-section"><span class="f1-accent">01 //</span> MATHEMATICAL TITLE SCENARIOS (ENTERING ABU DHABI)</div>', unsafe_allow_html=True)
    math_df = build_title_scenarios_summary(pre_race_df)
    st.dataframe(math_df, width="stretch", hide_index=True)

    # Section 2: Monte Carlo Title Probability
    st.markdown('<div class="f1-section"><span class="f1-accent">02 //</span> MONTE CARLO TITLE PROBABILITY & CONTENDERS</div>', unsafe_allow_html=True)

    contenders = sim_df[sim_df["Title Prob %"] > 0].copy()
    if contenders.empty:
        contenders = sim_df.head(5).copy()
    contenders["Color"] = contenders["Team"].map(TEAM_COLORS).fillna("#888888")

    col_prob, col_tbl = st.columns([1.05, 1.35])
    with col_prob:
        fig_prob = go.Figure(
            go.Bar(
                x=contenders["Driver"],
                y=contenders["Title Prob %"],
                marker_color=contenders["Color"],
                text=contenders["Title Prob %"].apply(lambda p: f"{p:.1f}%"),
                textposition="outside",
            )
        )
        fig_prob.update_layout(
            **PLOTLY_LAYOUT,
            title=f"WDC Title Probability ({n_sim:,} Runs · {sim_model_name})",
            yaxis_title="Title Probability (%)",
            height=390,
            yaxis_range=[0, max(contenders["Title Prob %"].max() * 1.2, 10)],
        )
        st.plotly_chart(fig_prob, width="stretch")

    with col_tbl:
        st.dataframe(
            sim_df.head(10),
            width="stretch",
            hide_index=True,
            height=402,
            column_config={
                "Grid": st.column_config.NumberColumn("Grid", format="P%.1f"),
                "Current Points": st.column_config.NumberColumn("Pts", format="%d"),
                "Gap to Leader": st.column_config.NumberColumn("Gap", format="-%d"),
                "Expected Finish": st.column_config.NumberColumn("Exp Pos", format="P%.2f"),
                "Race Win %": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                "Podium %": st.column_config.NumberColumn("Podium %", format="%.1f%%"),
                "DNF %": st.column_config.NumberColumn("DNF %", format="%.1f%%"),
                "Title Prob %": st.column_config.ProgressColumn(
                    "Title Prob %",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )

    # Section 3: Simulated Finish Position Distribution
    st.markdown('<div class="f1-section"><span class="f1-accent">03 //</span> SIMULATED FINISH POSITION DISTRIBUTION</div>', unsafe_allow_html=True)

    top_dist_drivers = sim_df.head(6)["Driver"].tolist()
    dist_long = pos_df[top_dist_drivers].melt(var_name="Driver", value_name="Simulated Finish")
    team_map = sim_df.set_index("Driver")["Team"].to_dict()
    dist_long["Team"] = dist_long["Driver"].map(team_map)

    fig_dist = px.violin(
        dist_long,
        x="Driver",
        y="Simulated Finish",
        color="Team",
        color_discrete_map=TEAM_COLORS,
        box=True,
        points=False,
        title="Distribution of Simulated Finishing Positions (Top 6 Contenders)",
    )
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        yaxis_autorange="reversed",
        yaxis_title="Finishing Position",
        showlegend=False,
    )
    st.plotly_chart(fig_dist, width="stretch")


# ===========================================================================
# PAGE 05 — ABOUT THE MODEL
# ===========================================================================
elif page == "05 — About the Model":
    st.markdown(
        """
        <div class="f1-hero">
            <div class="f1-hero-kicker">TRANSPARENCY & DOCUMENTATION</div>
            <h1 class="f1-hero-title">ABOUT THE MODEL</h1>
            <div class="f1-hero-sub">
                How the 2025 season data is collected, engineered, validated, and simulated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        ### 1. Dataset & Sources
        - **Source:** Official Formula 1 timing & race results via the `fastf1` Python library, cached locally in `data/cache/`.
        - **Scope:** 2025 Formula 1 World Championship (Rounds 1–23 up to Qatar GP, plus official pre-race grid for Round 24 Abu Dhabi).

        ### 2. Features Used
        Every feature is computed **strictly from past rounds** (`Round < t`) to prevent data leakage:
        - `grid_pos`: Starting grid position for the race.
        - `driver_avg_pos`: Driver's rolling mean finishing position across all prior rounds.
        - `rolling_3_pos`: Driver's recent 3-race rolling mean finishing position (captures mid-season upgrades and momentum).
        - `team_avg_pos`: Constructor's rolling mean finishing position across prior rounds.
        - `dnf_rate`: Driver's historical Did-Not-Finish rate across prior rounds.

        ### 3. Methodology
        - **Problem Framing:** Finish position regression (predicting exact finishing position `1..20`).
        - **Models:** `RandomForestRegressor` (300 trees, max depth 6) vs. `XGBRegressor` (300 trees, max depth 4, lr 0.05).
        - **Validation:** Temporal 75% / 25% split — earlier rounds train the model, later rounds test it.
        - **Simulation:** Monte Carlo sampling ($N$ runs) combining Gaussian pace variance ($\sigma$), explicit stochastic DNF events based on each driver's `dnf_rate`, and official FIA tie-breaker rules (total points, then total season wins).

        ### 4. Limitations
        - Does not yet incorporate live weather radar or compound-by-compound tire stint strategies.
        - Assumes independent DNF probabilities rather than multi-car Turn 1 collisions.
        """
    )
