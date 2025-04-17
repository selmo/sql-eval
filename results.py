import streamlit as st
import pandas as pd
import glob
import os

st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")

@st.cache_data
def load_data():
    """Load all CSV files in the current directory and add metadata columns."""
    rows = []
    for path in glob.glob("*.csv"):
        base = os.path.basename(path)
        try:
            engine, style, _ = base.split("_")  # e.g. gemma3_advanced.csv
        except ValueError:
            # Skip files that don't match the expected naming scheme
            continue
        df = pd.read_csv(path)
        df["engine"] = engine
        df["style"] = style
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# Load & pre‑process data
# -----------------------------------------------------------------------------

df = load_data()

if df.empty:
    st.warning("No CSV files found in the current directory. Place the evaluation CSVs next to this script and refresh.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar – interactive filters
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    engines = st.multiselect("Engine", sorted(df["engine"].unique()), default=sorted(df["engine"].unique()))
    styles = st.multiselect("Prompt style", sorted(df["style"].unique()), default=sorted(df["style"].unique()))
    categories = st.multiselect(
        "Query category", sorted(df["query_category"].unique()), default=sorted(df["query_category"].unique())
    )
    cols = st.multiselect(
        "Columns to show in raw data",
        list(df.columns),
        default=[
            "db_name",
            "query_category",
            "question",
            "generated_query",
            "correct",
            "latency_seconds",
        ],
    )

# Apply filters
filtered = df[
    df["engine"].isin(engines)
    & df["style"].isin(styles)
    & df["query_category"].isin(categories)
]

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------

st.title("LLM Evaluation Dashboard")
st.caption("Accuracy and latency across engines and prompt styles")

# Summary table – aggregated metrics
summary = (
    filtered.groupby(["engine", "style"]).agg(
        total=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        avg_latency=("latency_seconds", "mean"),
    ).reset_index()
)

st.subheader("Summary by Engine & Prompt Style")
st.dataframe(summary.style.format({"accuracy": "{:.1%}", "avg_latency": "{:.2f} s"}))

# Accuracy bar chart
st.subheader("Accuracy (%)")
chart_df = summary.pivot(index="engine", columns="style", values="accuracy") * 100
st.bar_chart(chart_df)

# Latency bar chart
st.subheader("Average Latency (s)")
lat_df = summary.pivot(index="engine", columns="style", values="avg_latency")
st.bar_chart(lat_df)

# Raw data
with st.expander("Show raw data"):
    st.dataframe(filtered[cols])
