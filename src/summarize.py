import streamlit as st
import nltk
from collections import Counter
import re

# Ensure required NLTK tokenizer resources are downloaded quietly on boot
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_extractive(text: str, sentence_count: int = 3) -> str:
    """Generates a fast extractive summary using TextRank (Ultra-low RAM)."""
    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Extractive summarization error: {e}"

def get_abstractive_summarizer():
    """Stub loader maintained for app.py compatibility without loading heavy PyTorch weights."""
    return None

def summarize_abstractive(summarizer_pipe, text: str, max_len: int = 130, min_len: int = 30) -> str:
    """
    Generates a lightweight key-sentence abstraction without loading PyTorch model weights.
    Prevents Out-Of-Memory (OOM) server crashes on Streamlit Cloud.
    """
    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."
    
    try:
        # Frequency-based key sentence extraction
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 2:
            return text
            
        words = re.findall(r'\w+', text.lower())
        stopwords = set(["the", "and", "a", "of", "to", "in", "is", "was", "for", "patient", "with", "on", "at", "by", "an", "be"])
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        freq = Counter(words)
        sentence_scores = {}
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if word in freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq[word]
                    
        sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        return " ".join(sorted_sentences[:2])
    except Exception as e:
        return f"Abstractive summarization error: {e}"
