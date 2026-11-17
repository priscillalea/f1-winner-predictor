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

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0B0D10",
    plot_bgcolor="#14171D",
    font=dict(color="#E5E7EB", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=dict(gridcolor="#232832", zerolinecolor="#232832"),
    yaxis=dict(gridcolor="#232832", zerolinecolor="#232832"),
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
    gap_str = "CHampionship Leader" if gap == 0 else f"-{gap:.0f} PTS TO LEADER"
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
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #232832; margin-bottom: 1rem;">
            <div style="color:#E10600; font-family:'Barlow Condensed',sans-serif; font-weight:800; font-size:1.5rem; letter-spacing:0.08em;">
                F1 ANALYTICS
            </div>
            <div style="color:#9CA3AF; font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase;">
                2025 Season · Data & Prediction
            </div>
        </div>
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

    st.markdown("---")
    st.caption(
        f"**Dataset:** {df['Round'].nunique()} Race Weekends · {len(df)} Driver Entries\n\n"
        f"**Best Model:** {ml_res['best_name']}\n\n"
        f"**Test MAE:** {ml_res['eval_df'].loc[ml_res['eval_df']['Model'] == ml_res['best_name'], 'Test MAE'].values[0]:.2f} pos"
    )


# ===========================================================================
# PAGE 01 — SEASON OVERVIEW
# ===========================================================================
if page == "01 — Season Overview":
    st.markdown(
        """
        <div class="f1-hero">
            <div class="f1-hero-kicker">2025 SEASON · DATA & PREDICTION</div>
            <h1 class="f1-hero-title">F1 ANALYTICS & CHAMPIONSHIP PREDICTOR</h1>
            <div class="f1-hero-sub">
                From race data to predictions. Explore the 2025 season, compare machine learning models,
                and simulate championship scenarios entering the season finale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown('<div class="f1-section"><span class="f1-accent">01 //</span> TITLE CONTENDERS — TOP 3 WDC</div>', unsafe_allow_html=True)
    leader_pts = float(wdc.iloc[0]["Points"])
    p1_col, p2_col, p3_col = st.columns(3)
    for idx, col in enumerate([p1_col, p2_col, p3_col]):
        with col:
            driver_broadcast_card(idx + 1, wdc.iloc[idx], leader_pts)

    st.markdown('<div class="f1-section"><span class="f1-accent">02 //</span> 2025 CHAMPIONSHIP STANDINGS</div>', unsafe_allow_html=True)

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
        st.plotly_chart(fig_wdc, use_container_width=True)

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
        st.plotly_chart(fig_wcc, use_container_width=True)

    st.markdown('<div class="f1-section"><span class="f1-accent">03 //</span> CUMULATIVE POINTS EVOLUTION & FULL TABLE</div>', unsafe_allow_html=True)

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
    st.plotly_chart(fig_cum, use_container_width=True)

    with st.expander("View Full 2025 Driver Standings Table (Broadcast View)", expanded=False):
        st.dataframe(
            wdc[["Rank", "Abbreviation", "FullName", "TeamName", "Wins", "Podiums", "AvgFinish", "Points"]],
            use_container_width=True,
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
        st.plotly_chart(fig_scatter, use_container_width=True)

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
        st.plotly_chart(fig_dnf, use_container_width=True)

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
    st.plotly_chart(fig_h2h, use_container_width=True)


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

    col_tbl, col_imp = st.columns([1, 1.2])
    with col_tbl:
        st.dataframe(
            ml_res["eval_df"],
            use_container_width=True,
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
        st.caption(
            "**Features included:** `grid_pos`, `driver_avg_pos` (season rolling), `rolling_3_pos` (recent 3-race form), "
            "`team_avg_pos`, and `dnf_rate`."
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
                marker_color="#E10600",
                text=imp_df["Importance"].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
            )
        )
        fig_imp.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Feature Importance — {ml_res['best_name']}",
            height=300,
            xaxis_title="Relative Importance",
            xaxis_range=[0, imp_df["Importance"].max() * 1.22],
        )
        st.plotly_chart(fig_imp, use_container_width=True)

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
        st.plotly_chart(fig_pred, use_container_width=True)

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
        st.plotly_chart(fig_res, use_container_width=True)

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
        use_container_width=True,
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
    st.dataframe(math_df, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_tbl:
        st.dataframe(
            sim_df.head(10),
            use_container_width=True,
            hide_index=True,
            height=390,
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
    st.plotly_chart(fig_dist, use_container_width=True)


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
        """
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
