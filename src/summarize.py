import streamlit as st
import nltk

# Ensure required NLTK tokenizer resources are downloaded quietly on runtime
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_extractive(text: str, sentence_count: int = 3) -> str:
    """Generates an extractive summary using TextRank."""
    if not text or len(text.strip()) == 0:
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Extractive summarization error: {e}"

@st.cache_resource(show_spinner=False)
def get_abstractive_summarizer():
    """Initializes and caches the Hugging Face abstractive summarization pipeline."""
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        framework="pt"
    )

def summarize_abstractive(summarizer_pipe, text: str, max_len: int = 130, min_len: int = 30) -> str:
    """Generates an abstractive summary using DistilBART."""
    if not text or len(text.strip()) == 0:
        return ""
    try:
        # Truncate input to max 1024 tokens to fit DistilBART position embeddings
        truncated_text = text[:1024]
        res = summarizer_pipe(truncated_text, max_length=max_len, min_length=min_len, do_sample=False)
        return res[0]['summary_text']
    except Exception as e:
        return f"Abstractive summarization error: {e}"
