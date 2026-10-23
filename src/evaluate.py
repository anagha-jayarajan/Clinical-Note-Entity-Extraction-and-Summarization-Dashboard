import json
import pandas as pd
from rouge_score import rouge_scorer

from src.extract import load_nlp_pipeline, extract_entities
from src.summarize import summarize_extractive, get_abstractive_summarizer, summarize_abstractive

GOLD_PATH = "data/annotations/gold_sample.json"

def evaluate_ner_performance(nlp, eval_data: list):
    """Calculates strict Precision, Recall, and F1-score against hand-annotated entities."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for item in eval_data:
        gold_set = set((e['text'].lower(), e['label']) for e in item['gold_entities'])
        
        preds = extract_entities(nlp, item['transcription'])
        pred_set = set((e['text'].lower(), e['label']) for e in preds)

        tp = len(gold_set.intersection(pred_set))
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1-Score": round(f1, 4)}

def evaluate_summarization_rouge(abs_pipe, eval_data: list):
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores for TextRank and DistilBART."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ext_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    abs_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for item in eval_data:
        reference = item['description']
        ext_summary = summarize_extractive(item['transcription'])
        abs_summary = summarize_abstractive(abs_pipe, item['transcription'])

        # Extractive evaluation
        ext_res = scorer.score(reference, ext_summary)
        for k in ext_scores:
            ext_scores[k].append(ext_res[k].fmeasure)

        # Abstractive evaluation
        abs_res = scorer.score(reference, abs_summary)
        for k in abs_scores:
            abs_scores[k].append(abs_res[k].fmeasure)

    return {
        "Extractive (TextRank)": {k: round(pd.Series(v).mean(), 4) for k, v in ext_scores.items()},
        "Abstractive (DistilBART)": {k: round(pd.Series(v).mean(), 4) for k, v in abs_scores.items()}
    }

def run_full_evaluation():
    print("Loading test data and models...")
    with open(GOLD_PATH, 'r') as f:
        eval_data = json.load(f)

    nlp = load_nlp_pipeline()
    abs_pipe = get_abstractive_summarizer()

    print("\n==========================================")
    print("1. CLINICAL NER EVALUATION (scispaCy)")
    print("==========================================")
    ner_metrics = evaluate_ner_performance(nlp, eval_data)
    for metric, val in ner_metrics.items():
        print(f"  {metric:<12}: {val}")

    print("\n==========================================")
    print("2. SUMMARIZATION BENCHMARKING (ROUGE)")
    print("==========================================")
    rouge_metrics = evaluate_summarization_rouge(abs_pipe, eval_data)
    rouge_df = pd.DataFrame(rouge_metrics).T
    print(rouge_df)

if __name__ == "__main__":
    run_full_evaluation()