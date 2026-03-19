import os
import pandas as pd
import streamlit as st

# Optional Groq import
try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(
    page_title="AI-Assisted Advertiser Performance Intelligence Engine",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
[data-testid="stMetric"] {
    background-color: #f7f9fc;
    border: 1px solid #e6eaf2;
    padding: 14px;
    border-radius: 14px;
}
h1, h2, h3 {
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

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
# GROQ CLIENT
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
client = None

if Groq is not None and groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except Exception:
        client = None

# -----------------------------
# GROQ INSIGHT FUNCTION
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
        return "Live AI insight is unavailable right now. Showing rule-based recommendation logic elsewhere in the dashboard."

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
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a precise marketing analytics expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Live AI insight is temporarily unavailable due to API quota or configuration limits. Please refer to the recommendation narratives section."

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
st.title("AI-Assisted Advertiser Performance Intelligence Engine")
st.markdown("""
A decision-support dashboard that analyzes campaign performance using **CTR, CVR, CPA, and ROI**,
detects anomalies, and recommends **budget reallocation actions** to improve advertiser efficiency.
""")

# -----------------------------
# KPI SUMMARY
# -----------------------------
total_campaigns = len(filtered_score)
avg_roi = round(filtered_score["avg_roi"].mean(), 2) if total_campaigns > 0 else 0
avg_cpa = round(filtered_score["avg_cpa"].mean(), 2) if total_campaigns > 0 else 0
avg_ctr = round(filtered_score["avg_ctr"].mean(), 4) if total_campaigns > 0 else 0
avg_cvr = round(filtered_score["avg_cvr"].mean(), 4) if total_campaigns > 0 else 0

healthy_count = (filtered_score["campaign_health"] == "Healthy").sum() if "campaign_health" in filtered_score.columns else 0
watchlist_count = (filtered_score["campaign_health"] == "Watchlist").sum() if "campaign_health" in filtered_score.columns else 0
critical_count = (filtered_score["campaign_health"] == "Critical").sum() if "campaign_health" in filtered_score.columns else 0

increase_count = filtered_optimizer["budget_action"].astype(str).str.contains("Increase", na=False).sum() if "budget_action" in filtered_optimizer.columns else 0
decrease_count = filtered_optimizer["budget_action"].astype(str).str.contains("Decrease", na=False).sum() if "budget_action" in filtered_optimizer.columns else 0
monitor_count = filtered_optimizer["budget_action"].astype(str).str.contains("Maintain", na=False).sum() if "budget_action" in filtered_optimizer.columns else 0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Campaigns", total_campaigns)
m2.metric("Avg ROI", avg_roi)
m3.metric("Avg CPA", avg_cpa)
m4.metric("Healthy", healthy_count)
m5.metric("Watchlist", watchlist_count)
m6.metric("Critical", critical_count)

st.success(
    f"Executive takeaway: {increase_count} campaign(s) are recommended for budget increases, "
    f"{decrease_count} should be reduced, and {monitor_count} require monitoring."
)

st.divider()

# -----------------------------
# CHARTS
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Top Campaigns by Performance Score")
    if len(filtered_score) > 0:
        perf_chart = (
            filtered_score[["campaign_name", "performance_score"]]
            .sort_values("performance_score", ascending=False)
            .head(8)
            .set_index("campaign_name")
        )
        st.bar_chart(perf_chart)

with c2:
    st.subheader("Average ROI by Campaign")
    if len(filtered_score) > 0:
        roi_chart = (
            filtered_score[["campaign_name", "avg_roi"]]
            .sort_values("avg_roi", ascending=False)
            .head(8)
            .set_index("campaign_name")
        )
        st.bar_chart(roi_chart)

st.divider()

# -----------------------------
# TOP / BOTTOM PERFORMERS
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Top Performers")
    if len(filtered_score) > 0:
        top_df = (
            filtered_score[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]]
            .sort_values("performance_score", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        st.dataframe(top_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Bottom Performers")
    if len(filtered_score) > 0:
        bottom_df = (
            filtered_score[["campaign_name", "performance_score", "avg_roi", "avg_cpa", "campaign_health"]]
            .sort_values("performance_score", ascending=True)
            .head(5)
            .reset_index(drop=True)
        )
        st.dataframe(bottom_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# OPTIMIZATION ACTIONS
# -----------------------------
st.subheader("Optimization Actions")

if len(filtered_optimizer) > 0:
    action_df = filtered_optimizer[[
        "campaign_name", "channel", "avg_roi", "avg_cpa", "avg_cvr",
        "performance_score", "campaign_health", "budget_action", "recommendation_reason"
    ]].copy()

    action_df = action_df.sort_values("performance_score", ascending=False).reset_index(drop=True)
    st.dataframe(action_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# RECOMMENDATION NARRATIVES
# -----------------------------
st.subheader("Recommendation Narratives")

if len(filtered_recommendation) > 0:
    rec_view = filtered_recommendation[[
        "campaign_name", "campaign_health", "budget_action", "ai_recommendation"
    ]].copy()

    for _, row in rec_view.iterrows():
        with st.expander(f"{row['campaign_name']} — {row['budget_action']}"):
            st.markdown(f"**Campaign Health:** {row['campaign_health']}")
            st.write(row["ai_recommendation"])

st.divider()

# -----------------------------
# OPTIONAL LIVE AI SECTION
# -----------------------------
show_llm = st.checkbox("Show live AI campaign insights")

if show_llm:
    st.subheader("Live AI Insights")
    if client is None:
        st.info("Live AI insights are not available because Groq is not configured.")
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
        ].copy().head(3)

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

st.divider()

# -----------------------------
# RISK ALERTS
# -----------------------------
st.subheader("Performance Risk Alerts")

if len(filtered_anomaly) > 0:
    risk_cols = [
        "date", "campaign_name", "spend", "ctr_sql", "cvr_sql",
        "roi_sql", "python_anomaly_score", "python_anomaly_label", "anomaly_flag"
    ]
    available_cols = [col for col in risk_cols if col in filtered_anomaly.columns]

    sort_cols = [col for col in ["python_anomaly_score", "date"] if col in filtered_anomaly.columns]

    risk_df = (
        filtered_anomaly[available_cols]
        .sort_values(by=sort_cols, ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# TREND VIEW
# -----------------------------
st.subheader("Performance Trend")

if len(filtered_anomaly) > 0 and "date" in filtered_anomaly.columns:
    metric_options = [
        col for col in ["spend", "ctr_sql", "cvr_sql", "roi_sql", "python_anomaly_score"]
        if col in filtered_anomaly.columns
    ]

    if len(metric_options) > 0:
        metric_choice = st.selectbox("Select a metric", metric_options)

        trend_df = filtered_anomaly.copy()

        if selected_campaign == "All":
            trend_df = trend_df.groupby("date", as_index=False)[metric_choice].mean()

        chart_df = trend_df.set_index("date")[[metric_choice]]
        st.line_chart(chart_df)

st.divider()

# -----------------------------
# SUMMARY
# -----------------------------
st.subheader("Project Summary")
st.markdown("""
This prototype demonstrates how campaign performance data can be transformed into an **AI-assisted optimization engine**
using a combination of **SQL analytics, Python-based campaign scoring, anomaly detection, and recommendation logic**.

It helps identify:
- which campaigns are performing efficiently,
- where risk signals are emerging,
- and where budget should be scaled, maintained, or reduced.
""")
