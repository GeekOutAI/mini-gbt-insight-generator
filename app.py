import streamlit as st
import pandas as pd

from pipeline import load_data, preprocess, run_pipeline, export_scored_csv

st.set_page_config(page_title="Mini-GPT Customer Insight Generator", layout="wide")
st.title("🧠 Mini-GPT Customer Insight Generator")
st.write("Upload a CSV of customer comments or product reviews. The app will score sentiment, extract themes, optionally cluster topics, and generate an executive summary.")

with st.sidebar:
    st.header("Options")
    do_cluster = st.toggle("Cluster themes (UMAP + HDBSCAN)", value=False)
    summary_method = st.radio(
        "Summarization method",
        options=["open_source", "openai"],
        index=0,
        help="Open-source uses BART. 'openai' requires OPENAI_API_KEY."
    )
    st.caption("Tip: Start with open_source. Switch to 'openai' if you want a punchier summary and have a key set.")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    try:
        df = load_data(uploaded)
        df = preprocess(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    with st.spinner("Running analysis... (sentiment, keywords, summary)"):
        try:
            scored_df, summary, metrics = run_pipeline(df, do_cluster=do_cluster, summary_method=summary_method)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Comments", metrics["total"])
    with c2:
        st.metric("Positive %", f"{metrics['pos_pct']:.1f}%")
    with c3:
        st.metric("Negative %", f"{metrics['neg_pct']:.1f}%")

    st.subheader("Executive Summary")
    st.write(summary)

    st.subheader("Sample of Scored Comments")
    st.dataframe(scored_df[["comment","sent_label","sent_score","keywords","topic"]].head(25))

    st.download_button(
        "⬇️ Download Scored CSV",
        data=export_scored_csv(scored_df),
        file_name="scored_comments.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("DistilBERT sentiment · KeyBERT themes · optional UMAP+HDBSCAN clustering · BART/OpenAI summary.")
