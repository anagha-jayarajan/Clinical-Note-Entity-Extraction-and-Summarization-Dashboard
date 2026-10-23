import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

def summarize_extractive(text: str, sentence_count: int = 2) -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sentences = summarizer(parser.document, sentence_count)
        return " ".join([str(s) for s in sentences])
    except Exception:
        return text

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def get_abstractive_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline(
            task="summarization",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
    except Exception as e:
        # Fallback to standard facebook/bart-large-cnn if model resolution fails
        return pipeline(
            task="summarization", 
            model="facebook/bart-large-cnn", 
            device=-1
        )

def summarize_abstractive(pipe, text: str) -> str:
    try:
        out = pipe(text, max_length=60, min_length=15, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        return text

def verify_summary_faithfulness(asserted_entities: list, summary_text: str) -> list:
    summary_lower = summary_text.lower()
    return [e for e in asserted_entities if e.lower() not in summary_lower]
