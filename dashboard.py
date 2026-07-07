import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="PRANA Dashboard",
    layout="wide"
)

st.title("🩺 PRANA AI TRIAGE DASHBOARD")
st_autorefresh(
    interval=5000,
    key="prana_refresh"
)

# ---------------------------------
# Load Data
# ---------------------------------

conn = sqlite3.connect("prana_records.db")

df = pd.read_sql_query(
    """
    SELECT *
    FROM triage_reports
    ORDER BY id DESC
    """,
    conn
)

conn.close()

# ---------------------------------
# Metrics
# ---------------------------------

st.subheader("System Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Reports",
        len(df)
    )

with col2:
    st.metric(
        "Critical",
        len(df[df["overall"] == "CRITICAL"])
    )

with col3:
    st.metric(
        "High",
        len(df[df["overall"] == "HIGH"])
    )

with col4:
    st.metric(
        "Normal",
        len(df[df["overall"] == "NORMAL"])
    )
    
with col5:
    st.metric(
        "Moderate",
        len(df[df["overall"] == "MODERATE"])
    )
# ---------------------------------
# Severity Chart
# ---------------------------------

st.subheader("Severity Distribution")

if len(df) > 0:

    severity_counts = (
        df["overall"]
        .value_counts()
    )

    fig = px.pie(
        names=severity_counts.index,
        values=severity_counts.values,
        title="Patient Severity Distribution"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# ---------------------------------
# ECG Confidence Statistics
# ---------------------------------

st.subheader(
    "Confidence Distribution"
)

fig_conf = px.histogram(
    df,
    x="ecg_confidence",
    nbins=20,
    title="ECG Prediction Confidence"
)

st.plotly_chart(
    fig_conf,
    width="stretch"
)

st.subheader("📈 ECG Confidence Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Confidence",
        f"{df['ecg_confidence'].mean():.1f}%"
    )

with col2:
    st.metric(
        "Maximum Confidence",
        f"{df['ecg_confidence'].max():.1f}%"
    )

with col3:
    st.metric(
        "Minimum Confidence",
        f"{df['ecg_confidence'].min():.1f}%"
    )

with col4:
    st.metric(
        "Median Confidence",
        f"{df['ecg_confidence'].median():.1f}%"
    )

# ---------------------------------
# Critical Cases
# ---------------------------------

st.subheader("🚨 Critical / High Priority Cases")

critical_df = (
    df[
        df["overall"].isin(
            ["CRITICAL", "HIGH"]
        )
    ]
    .head(20)
)

st.dataframe(
    critical_df[
        [
            "patient_id",
            "ecg_status",
            "ecg_confidence",
            "xray_status",
            "overall",
            "action"
        ]
    ],
    width="stretch"
)

# ------------------------------
# Full Table
# ---------------------------------

severity_filter = st.selectbox(
    "Filter by Severity",
    ["ALL", "CRITICAL", "HIGH", "MODERATE", "NORMAL"]
)

if severity_filter != "ALL":
    filtered_df = df[
        df["overall"] == severity_filter
    ]
else:
    filtered_df = df

patient_search = st.text_input(
    "Search Patient ID"
)

if patient_search:
    filtered_df = filtered_df[
        filtered_df["patient_id"]
        .str.contains(
            patient_search,
            case=False
        )
    ]

st.subheader("All Triage Reports")

st.dataframe(
    filtered_df[
        [
            "patient_id",
            "timestamp",
            "ecg_status",
            "ecg_confidence",
            "xray_status",
            "overall",
            "action"
        ]
    ],
    width="stretch"
)

# ---------------------------------
# Latest Reports
# ---------------------------------

st.subheader("Latest 10 Reports")

st.dataframe(
    df.head(10),
    width="stretch"
)