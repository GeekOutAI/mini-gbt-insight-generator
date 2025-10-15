import re
from typing import List
import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap
import hdbscan

from models import score_sentiment, extract_keywords, summarize

# -------- Data loading / cleaning --------
def load_data(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    if "comment" not in df.columns:
        for c in ["text", "review", "message", "body", "comments", "content"]:
            if c in df.columns:
                df = df.rename(columns={c: "comment"})
                break
    if "comment" not in df.columns:
        raise ValueError("No text column found. Include 'comment' or one of: text, review, message, body.")
    df = df.dropna(subset=["comment"]).reset_index(drop=True)
    return df

def _clean_text(s: str) -> str:
    s = re.sub(r"http\S+", "", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean"] = df["comment"].astype(str).apply(_clean_text)
    return df

# -------- Topic clustering (optional) --------
def cluster_topics(texts: List[str], min_cluster_size: int = 10, max_features: int = 5000, random_state: int = 42):
    if len(texts) < max(20, min_cluster_size * 2):
        return np.full(len(texts), -1, dtype=int)
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(texts)
    emb = umap.UMAP(
        n_neighbors=15, min_dist=0.0, metric="cosine", random_state=random_state
    ).fit_transform(X)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(emb)
    return labels

# -------- Executive summary text builder --------
def build_summary_corpus(df: pd.DataFrame, pos_n: int = 10, neg_n: int = 10) -> str:
    neg = df[df["sent_label"] == "Negative"]["clean"].head(80).tolist()
    pos = df[df["sent_label"] == "Positive"]["clean"].head(80).tolist()
    corpus = (
        "Positive highlights:\n- " + "\n- ".join(pos[:pos_n]) +
        "\n\nNegative pain points:\n- " + "\n- ".join(neg[:neg_n])
    )
    return corpus

def run_pipeline(df: pd.DataFrame, do_cluster: bool = False, summary_method: str = "open_source"):
    df["sent_label"], df["sent_score"] = score_sentiment(df["clean"])
    df["keywords"] = extract_keywords(df["clean"], top_k=3)
    df["topic"] = cluster_topics(df["clean"]) if do_cluster else -1
    corpus = build_summary_corpus(df)
    summary = summarize(corpus, method=summary_method)
    total = len(df)
    pos_pct = (df["sent_label"] == "Positive").mean() * 100 if total else 0.0
    neg_pct = (df["sent_label"] == "Negative").mean() * 100 if total else 0.0
    metrics = {"total": total, "pos_pct": pos_pct, "neg_pct": neg_pct}
    return df, summary, metrics

# -------- Export --------
def export_scored_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
