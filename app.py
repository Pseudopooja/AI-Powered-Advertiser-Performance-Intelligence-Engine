import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Advertiser Performance Intelligence Engine", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    score_df = pd.read_csv("campaign_scoring_output.csv")
    anomaly_df = pd.read_csv("anomaly_analysis_output.csv")
    optimizer_df = pd.read_csv("budget_optimization_output.csv")
    recommendation_df = pd.read_csv("ai_recommendations_output.csv")
    return score_df, anomaly_df, optimizer_df, recommendation_df

score_df, anomaly_df, optimizer_df, recommendation_df = load_data()

# Date handling
if "date" in anomaly_df.columns:
    anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

campaign_options = ["All"] + sorted(score_df["campaign_name"].dropna().unique().tolist())
health_options = ["All"] + sorted(score_df["campaign_health"].dropna().unique().tolist())
channel_options = ["All"] + sorted(score_df["channel"].dropna().unique().tolist())

selected_campaign = st.sidebar.selectbox("Campaign", campaign_options)
selected_health = st.sidebar.selectbox("Campaign Health", health_options)
selected_channel = st.sidebar.selectbox("Channel", channel_options)

filtered_score = score_df.copy()
filtered_optimizer = optimizer_df.copy()
filtered_recommendation = recommendation_df.copy()
filtered_anomaly = anomaly_df.copy()

if selected_campaign != "All":
    filtered_score = filtered_score[filtered_score["campaign_name"] == selected_campaign]
    filtered_optimizer = filtered_optimizer[filtered_optimizer["campaign_name"] == selected_campaign]
    filtered_recommendation = filtered_recommendation[filtered_recommendation["campaign_name"] == selected_campaign]
    filtered_anomaly = filtered_anomaly[filtered_anomaly["campaign_name"] == selected_campaign]

if selected_health != "All":
    filtered_score = filtered_score[filtered_score["campaign_health"] == selected_health]
    filtered_optimizer = filtered_optimizer[filtered_optimizer["campaign_health"] == selected_health]
    filtered_recommendation = filtered_recommendation[filtered_recommendation["campaign_health"] == selected_health]

if selected_channel != "All":
    filtered_score = filtered_score[filtered_score["channel"] == selected_channel]
    filtered_optimizer = filtered_optimizer[filtered_optimizer["channel"] == selected_channel]
    filtered_recommendation = filtered_recommendation[filtered_recommendation["channel"] == selected_channel]

# -----------------------------
# HEADER
# -----------------------------
st.title("AI-Powered Advertiser Performance Intelligence Engine")
st.markdown(
    "A campaign analytics and budget optimization dashboard using SQL and Python to evaluate CTR, CVR, CPA, ROI, detect anomalies, and recommend budget reallocation."
)

# -----------------------------
# KPI CARDS
# -----------------------------
total_campaigns = len(filtered_score)
avg_roi = round(filtered_score["avg_roi"].mean(), 2) if total_campaigns > 0 else 0
avg_cpa = round(filtered_score["avg_cpa"].mean(), 2) if total_campaigns > 0 else 0
avg_ctr = round(filtered_score["avg_ctr"].mean(), 4) if total_campaigns > 0 else 0
avg_cvr = round(filtered_score["avg_cvr"].mean(), 4) if total_campaigns > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Campaigns", total_campaigns)
col2.metric("Avg ROI", avg_roi)
col3.metric("Avg CPA", avg_cpa)
col4.metric("Avg CTR", avg_ctr)
col5.metric("Avg CVR", avg_cvr)

st.divider()

# -----------------------------
# CAMPAIGN LEADERBOARD
# -----------------------------
st.subheader("Campaign Leaderboard")

leaderboard_cols = [
    "campaign_name", "campaign_profile", "channel", "avg_ctr", "avg_cvr",
    "avg_cpa", "avg_roi", "anomaly_count", "performance_score", "campaign_health"
]
st.dataframe(
    filtered_score[leaderboard_cols].sort_values("performance_score", ascending=False),
    use_container_width=True
)

# -----------------------------
# TOP / BOTTOM PERFORMERS
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Top Performers")
    top_df = filtered_score.sort_values("performance_score", ascending=False).head(5)
    st.dataframe(
        top_df[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]],
        use_container_width=True
    )

with c2:
    st.subheader("Bottom Performers")
    bottom_df = filtered_score.sort_values("performance_score", ascending=True).head(5)
    st.dataframe(
        bottom_df[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]],
        use_container_width=True
    )

st.divider()

# -----------------------------
# BUDGET RECOMMENDATIONS
# -----------------------------
st.subheader("Budget Reallocation Recommendations")

budget_cols = [
    "campaign_name", "campaign_profile", "channel", "avg_roi", "avg_cpa",
    "avg_cvr", "performance_score", "campaign_health", "budget_action", "recommendation_reason"
]
st.dataframe(
    filtered_optimizer[budget_cols].sort_values("performance_score", ascending=False),
    use_container_width=True
)

# -----------------------------
# AI RECOMMENDATIONS
# -----------------------------
st.subheader("AI-Style Narrative Recommendations")

if len(filtered_recommendation) > 0:
    for _, row in filtered_recommendation.iterrows():
        with st.expander(f"{row['campaign_name']} — {row['budget_action']}"):
            st.write(f"**Health:** {row['campaign_health']}")
            st.write(f"**Recommendation:** {row['ai_recommendation']}")
else:
    st.info("No recommendations available for the selected filters.")

st.divider()

# -----------------------------
# ANOMALY MONITOR
# -----------------------------
st.subheader("Anomaly Monitor")

if len(filtered_anomaly) > 0:
    anomaly_view = filtered_anomaly[
        [
            "date", "campaign_name", "spend", "ctr_sql", "cvr_sql",
            "roi_sql", "python_anomaly_score", "python_anomaly_label", "anomaly_flag"
        ]
    ].sort_values(["python_anomaly_score", "date"], ascending=[False, False])

    st.dataframe(anomaly_view.head(50), use_container_width=True)
else:
    st.info("No anomaly data available for the selected filters.")

# -----------------------------
# TIME TREND
# -----------------------------
st.subheader("Performance Trend Over Time")

if len(filtered_anomaly) > 0:
    metric_choice = st.selectbox(
        "Select a metric to visualize",
        ["spend", "ctr_sql", "cvr_sql", "roi_sql", "python_anomaly_score"]
    )

    trend_df = filtered_anomaly.copy()

    if selected_campaign == "All":
        trend_df = trend_df.groupby("date", as_index=False)[metric_choice].mean()

    chart_df = trend_df.set_index("date")[[metric_choice]]
    st.line_chart(chart_df)
else:
    st.info("No trend data available.")

st.divider()

# -----------------------------
# EXECUTIVE SUMMARY
# -----------------------------
st.subheader("Executive Summary")

healthy_count = (filtered_score["campaign_health"] == "Healthy").sum()
watchlist_count = (filtered_score["campaign_health"] == "Watchlist").sum()
critical_count = (filtered_score["campaign_health"] == "Critical").sum()
increase_count = filtered_optimizer["budget_action"].astype(str).str.contains("Increase", na=False).sum()
decrease_count = filtered_optimizer["budget_action"].astype(str).str.contains("Decrease", na=False).sum()

summary = f"""
- **{healthy_count}** campaigns are classified as Healthy.  
- **{watchlist_count}** campaigns are on the Watchlist.  
- **{critical_count}** campaigns are classified as Critical.  
- **{increase_count}** campaigns are recommended for budget increases.  
- **{decrease_count}** campaigns are recommended for budget reductions.  
- Average ROI across the filtered set is **{avg_roi}**.  
- Average CPA across the filtered set is **{avg_cpa}**.  

Overall, the engine highlights where spend can be scaled efficiently and where budget should be reduced to improve advertiser monetization efficiency.
"""
st.markdown(summary)
