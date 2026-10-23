import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

def summarize_extractive(text: str, sentence_count: int = 3) -> str:
    """Generates extractive summary using TextRank."""
    if not text or len(text.strip()) == 0:
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

@st.cache_resource
def get_abstractive_summarizer():
    """Initializes and caches the Hugging Face abstractive summarization pipeline."""
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt"
    )

def summarize_abstractive(summarizer_pipe, text: str, max_len: int = 130, min_len: int = 30) -> str:
    """Generates abstractive summary using DistilBART."""
    if not text or len(text.strip()) == 0:
        return ""
    # Truncate input text to prevent transformer token limit overflow
    truncated_text = text[:1024]
    res = summarizer_pipe(truncated_text, max_length=max_len, min_length=min_len, do_sample=False)
    return res[0]['summary_text']
