import os
import pandas as pd
import streamlit as st
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Advertiser Performance Intelligence Engine",
    layout="wide"
)

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

if "date" in anomaly_df.columns:
    anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])

# -----------------------------
# OPENAI CLIENT
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# LLM INSIGHT FUNCTION
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_llm_insight(
    campaign_name,
    avg_roi,
    avg_cpa,
    avg_cvr,
    avg_ctr,
    performance_score,
    campaign_health,
    budget_action
):
    if client is None:
        return "AI insights are currently unavailable because no API key is configured."

    prompt = f"""
You are a senior marketing performance analyst.

Analyze this campaign and provide:
1. A short diagnosis
2. One clear business recommendation

Keep it concise, practical, professional, and under 3 sentences.

Campaign Name: {campaign_name}
Average ROI: {avg_roi}
Average CPA: {avg_cpa}
Average CVR: {avg_cvr}
Average CTR: {avg_ctr}
Performance Score: {performance_score}
Campaign Health: {campaign_health}
Budget Action: {budget_action}
"""

    try:
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt
        )
        return response.output_text.strip()
    except Exception:
        return "AI insights are temporarily unavailable due to API quota or configuration limits. Please refer to the rule-based recommendations section above."

# -----------------------------
# SIDEBAR
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
    """
This dashboard evaluates campaign performance using SQL- and Python-driven logic to analyze
**CTR, CVR, CPA, ROI**, detect anomalies, and recommend budget reallocation actions.
"""
)

# -----------------------------
# KPI CARDS
# -----------------------------
total_campaigns = len(filtered_score)
avg_roi = round(filtered_score["avg_roi"].mean(), 2) if total_campaigns > 0 else 0
avg_cpa = round(filtered_score["avg_cpa"].mean(), 2) if total_campaigns > 0 else 0
avg_ctr = round(filtered_score["avg_ctr"].mean(), 4) if total_campaigns > 0 else 0
avg_cvr = round(filtered_score["avg_cvr"].mean(), 4) if total_campaigns > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Campaigns", total_campaigns)
c2.metric("Avg ROI", avg_roi)
c3.metric("Avg CPA", avg_cpa)
c4.metric("Avg CTR", avg_ctr)
c5.metric("Avg CVR", avg_cvr)

st.divider()

# -----------------------------
# CAMPAIGN LEADERBOARD
# -----------------------------
st.subheader("Campaign Leaderboard")

leaderboard_cols = [
    "campaign_name", "campaign_profile", "channel", "avg_ctr", "avg_cvr",
    "avg_cpa", "avg_roi", "anomaly_count", "performance_score", "campaign_health"
]

if len(filtered_score) > 0:
    st.dataframe(
        filtered_score[leaderboard_cols].sort_values("performance_score", ascending=False),
        use_container_width=True
    )
else:
    st.info("No campaign data available for the selected filters.")

st.divider()

# -----------------------------
# TOP / BOTTOM PERFORMERS
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Top Performers")
    if len(filtered_score) > 0:
        top_df = filtered_score.sort_values("performance_score", ascending=False).head(5)
        st.dataframe(
            top_df[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]],
            use_container_width=True
        )
    else:
        st.info("No top performers to display.")

with right:
    st.subheader("Bottom Performers")
    if len(filtered_score) > 0:
        bottom_df = filtered_score.sort_values("performance_score", ascending=True).head(5)
        st.dataframe(
            bottom_df[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]],
            use_container_width=True
        )
    else:
        st.info("No bottom performers to display.")

st.divider()

# -----------------------------
# BUDGET RECOMMENDATIONS
# -----------------------------
st.subheader("Budget Reallocation Recommendations")

budget_cols = [
    "campaign_name", "campaign_profile", "channel", "avg_roi", "avg_cpa",
    "avg_cvr", "performance_score", "campaign_health", "budget_action", "recommendation_reason"
]

if len(filtered_optimizer) > 0:
    st.dataframe(
        filtered_optimizer[budget_cols].sort_values("performance_score", ascending=False),
        use_container_width=True
    )
else:
    st.info("No budget recommendations available.")

st.divider()

# -----------------------------
# RULE-BASED RECOMMENDATIONS
# -----------------------------
st.subheader("Automated Recommendation Narratives")

if len(filtered_recommendation) > 0:
    for _, row in filtered_recommendation.iterrows():
        with st.expander(f"{row['campaign_name']} — {row['budget_action']}"):
            st.write(f"**Health:** {row['campaign_health']}")
            st.write(f"**Recommendation:** {row['ai_recommendation']}")
else:
    st.info("No automated recommendations available.")

st.divider()

# -----------------------------
# LLM-GENERATED AI INSIGHTS
# -----------------------------
st.subheader("LLM-Generated AI Insights")

if client is None:
    st.info("AI insights are currently disabled because no API key is configured.")
elif len(filtered_optimizer) > 0:
    llm_view = filtered_optimizer[
        [
            "campaign_name",
            "avg_roi",
            "avg_cpa",
            "avg_cvr",
            "avg_ctr",
            "performance_score",
            "campaign_health",
            "budget_action"
        ]
    ].copy().head(5)

    for _, row in llm_view.iterrows():
        with st.expander(f"{row['campaign_name']} — AI Insight"):
            insight = generate_llm_insight(
                campaign_name=row["campaign_name"],
                avg_roi=row["avg_roi"],
                avg_cpa=row["avg_cpa"],
                avg_cvr=row["avg_cvr"],
                avg_ctr=row["avg_ctr"],
                performance_score=row["performance_score"],
                campaign_health=row["campaign_health"],
                budget_action=row["budget_action"]
            )
            st.write(insight)
else:
    st.info("No campaigns available for AI insights.")

st.divider()

# -----------------------------
# ANOMALY MONITOR
# -----------------------------
st.subheader("Anomaly Monitor")

if len(filtered_anomaly) > 0:
    anomaly_cols = [
        "date", "campaign_name", "spend", "ctr_sql", "cvr_sql",
        "roi_sql", "python_anomaly_score", "python_anomaly_label", "anomaly_flag"
    ]
    available_anomaly_cols = [col for col in anomaly_cols if col in filtered_anomaly.columns]
    sort_cols = [col for col in ["python_anomaly_score", "date"] if col in filtered_anomaly.columns]

    st.dataframe(
        filtered_anomaly[available_anomaly_cols].sort_values(
            by=sort_cols,
            ascending=False
        ).head(50),
        use_container_width=True
    )
else:
    st.info("No anomaly data available.")

st.divider()

# -----------------------------
# TREND VIEW
# -----------------------------
st.subheader("Performance Trend Over Time")

if len(filtered_anomaly) > 0 and "date" in filtered_anomaly.columns:
    metric_options = [
        col for col in ["spend", "ctr_sql", "cvr_sql", "roi_sql", "python_anomaly_score"]
        if col in filtered_anomaly.columns
    ]

    if len(metric_options) > 0:
        metric_choice = st.selectbox("Select a metric to visualize", metric_options)

        trend_df = filtered_anomaly.copy()

        if selected_campaign == "All":
            trend_df = trend_df.groupby("date", as_index=False)[metric_choice].mean()

        chart_df = trend_df.set_index("date")[[metric_choice]]
        st.line_chart(chart_df)
    else:
        st.info("No numeric trend metrics available.")
else:
    st.info("No trend data available.")

st.divider()

# -----------------------------
# EXECUTIVE SUMMARY
# -----------------------------
st.subheader("Executive Summary")

healthy_count = (filtered_score["campaign_health"] == "Healthy").sum() if "campaign_health" in filtered_score.columns else 0
watchlist_count = (filtered_score["campaign_health"] == "Watchlist").sum() if "campaign_health" in filtered_score.columns else 0
critical_count = (filtered_score["campaign_health"] == "Critical").sum() if "campaign_health" in filtered_score.columns else 0

increase_count = filtered_optimizer["budget_action"].astype(str).str.contains("Increase", na=False).sum() if "budget_action" in filtered_optimizer.columns else 0
decrease_count = filtered_optimizer["budget_action"].astype(str).str.contains("Decrease", na=False).sum() if "budget_action" in filtered_optimizer.columns else 0

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
