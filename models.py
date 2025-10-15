import os
from typing import List, Tuple, Iterable

from transformers import pipeline as hf_pipeline
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ------- Sentiment -------
_sentiment = None
def get_sentiment_pipeline():
    global _sentiment
    if _sentiment is None:
        _sentiment = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment

def score_sentiment(texts: Iterable[str]) -> Tuple[List[str], List[float]]:
    nlp = get_sentiment_pipeline()
    out = nlp(list(texts))
    labels = [o["label"].title() for o in out]
    scores = [float(o["score"]) for o in out]
    return labels, scores

# ------- Keywords / Themes -------
_kw_model = None
def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model

def extract_keywords(texts: Iterable[str], top_k: int = 3) -> List[List[str]]:
    kw = get_kw_model()
    results: List[List[str]] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            results.append([])
            continue
        pairs = kw.extract_keywords(
            t, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_k
        )
        results.append([p[0] for p in pairs])
    return results

# ------- Summarization (BART open-source or OpenAI) -------
_bart_tokenizer = None
_bart_model = None

def _init_bart():
    global _bart_tokenizer, _bart_model
    if _bart_tokenizer is None or _bart_model is None:
        _bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        _bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize_open_source(text: str, min_len: int = 80, max_len: int = 220) -> str:
    _init_bart()
    inputs = _bart_tokenizer(text[:4000], return_tensors="pt", truncation=True)
    summary_ids = _bart_model.generate(
        **inputs, max_length=max_len, min_length=min_len, do_sample=False
    )
    return _bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_openai(text: str, min_len_words: int = 120, max_len_words: int = 180) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI package not available. Use open-source summarizer or install openai.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an expert CX analyst. Summarize the top themes, customer sentiment, and actionable recommendations "
        "from the bullets below. Be concise ({}–{} words). Start with 3 bullets: Wins, Risks, Actions. "
        "Then a 1-sentence CTA.\n\n{}"
    ).format(min_len_words, max_len_words, text)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def summarize(text: str, method: str = "open_source") -> str:
    return summarize_openai(text) if method == "openai" else summarize_open_source(text)
