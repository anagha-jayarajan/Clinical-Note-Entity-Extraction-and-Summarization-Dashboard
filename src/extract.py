import spacy
from negspacy.negation import Negex

def load_nlp_pipeline():
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
    except Exception:
        nlp = spacy.load("en_core_web_sm")
    
    if "negex" not in nlp.pipe_names:
        nlp.add_pipe("negex", config={"ent_types": ["DISEASE", "CHEMICAL"]}, last=True)
    return nlp

def extract_entities(nlp, text: str) -> list:
    doc = nlp(text)
    extracted = []
    
    for ent in doc.ents:
        is_negated = 1 if getattr(ent._, "negex", False) else 0
        cui = f"C{abs(hash(ent.text.lower())) % 10000000:07d}"
        
        extracted.append({
            "text": ent.text,
            "label": ent.label_,
            "negated": is_negated,
            "cui": cui,
            "canonical_name": ent.text.title()
        })
    return extracted

if __name__ == "__main__":
    nlp = load_nlp_pipeline()
    sample = "Patient denies chest pain. Diagnosed with hypertension and prescribed metformin."
    print(extract_entities(nlp, sample))