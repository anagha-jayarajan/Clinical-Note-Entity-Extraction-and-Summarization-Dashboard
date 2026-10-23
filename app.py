import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_and_clean_data
from src.extract import load_nlp_pipeline, extract_entities
from src.summarize import (
    summarize_extractive, 
    get_abstractive_summarizer, 
    summarize_abstractive, 
    verify_summary_faithfulness
)
from src.db import init_db, DB_NAME

st.set_page_config(page_title="Clinical NLP Dashboard", layout="wide")

@st.cache_resource
def initialize_pipeline_and_db():
    init_db()
    nlp = load_nlp_pipeline()
    abs_summarizer = get_abstractive_summarizer()
    df_cleaned = load_and_clean_data("data/raw/mtsamples.csv")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for _, row in df_cleaned.iterrows():
        cursor.execute(
            "INSERT INTO notes (medical_specialty, sample_name, transcription) VALUES (?, ?, ?)",
            (row['medical_specialty'], row['sample_name'], row['transcription'])
        )
        note_id = cursor.lastrowid
        
        ents = extract_entities(nlp, row['transcription'])
        for e in ents:
            cursor.execute(
                "INSERT INTO entities (note_id, text, label, negated, cui, canonical_name) VALUES (?, ?, ?, ?, ?, ?)",
                (note_id, e['text'], e['label'], e['negated'], e['cui'], e['canonical_name'])
            )
            
        ext_sum = summarize_extractive(row['transcription'])
        abs_sum = summarize_abstractive(abs_summarizer, row['transcription'])
        
        cursor.execute("INSERT INTO summaries (note_id, method, summary) VALUES (?, ?, ?)", (note_id, "Extractive (TextRank)", ext_sum))
        cursor.execute("INSERT INTO summaries (note_id, method, summary) VALUES (?, ?, ?)", (note_id, "Abstractive (DistilBART)", abs_sum))
        
    conn.commit()
    conn.close()

initialize_pipeline_and_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View:",
    ["Overview", "Note Explorer", "Summary View", "Analytics", "Search", "About / Ethics"]
)

if page == "Overview":
    st.title("Clinical Note Analytics Overview")
    conn = sqlite3.connect(DB_NAME)
    
    total_notes = pd.read_sql_query("SELECT COUNT(*) as c FROM notes", conn).iloc[0]['c']
    total_ents = pd.read_sql_query("SELECT COUNT(*) as c FROM entities", conn).iloc[0]['c']
    neg_ents = pd.read_sql_query("SELECT COUNT(*) as c FROM entities WHERE negated = 1", conn).iloc[0]['c']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Notes", total_notes)
    col2.metric("Total Extracted Entities", total_ents)
    col3.metric("Negated Share", f"{(neg_ents/total_ents*100) if total_ents else 0:.1f}%")
    
    st.subheader("Note Count by Specialty")
    spec_df = pd.read_sql_query("SELECT medical_specialty, COUNT(*) as count FROM notes GROUP BY medical_specialty", conn)
    st.bar_chart(spec_df.set_index("medical_specialty"))
    conn.close()

elif page == "Note Explorer":
    st.title("Clinical Note Explorer")
    conn = sqlite3.connect(DB_NAME)
    notes_df = pd.read_sql_query("SELECT note_id, sample_name FROM notes", conn)
    
    if not notes_df.empty:
        selected_id = st.selectbox("Select Note:", notes_df["note_id"], format_func=lambda x: f"Note #{x}: {notes_df[notes_df['note_id']==x]['sample_name'].values[0]}")
        note_data = pd.read_sql_query("SELECT * FROM notes WHERE note_id = ?", conn, params=(selected_id,)).iloc[0]
        entities_df = pd.read_sql_query("SELECT text, label, negated, cui, canonical_name FROM entities WHERE note_id = ?", conn, params=(selected_id,))
        
        st.write(f"**Specialty:** {note_data['medical_specialty']}")
        st.subheader("Transcription Text")
        st.write(note_data['transcription'])
        
        st.subheader("Extracted Medical Entities")
        st.dataframe(entities_df)
    conn.close()

elif page == "Summary View":
    st.title("Summarization & Faithfulness Comparison")
    conn = sqlite3.connect(DB_NAME)
    notes_df = pd.read_sql_query("SELECT note_id, sample_name FROM notes", conn)
    
    if not notes_df.empty:
        selected_id = st.selectbox("Select Note:", notes_df["note_id"], format_func=lambda x: f"Note #{x}: {notes_df[notes_df['note_id']==x]['sample_name'].values[0]}")
        note_data = pd.read_sql_query("SELECT * FROM notes WHERE note_id = ?", conn, params=(selected_id,)).iloc[0]
        summaries_df = pd.read_sql_query("SELECT method, summary FROM summaries WHERE note_id = ?", conn, params=(selected_id,))
        asserted_ents = pd.read_sql_query("SELECT text FROM entities WHERE note_id = ? AND negated = 0", conn, params=(selected_id,))['text'].tolist()
        
        st.subheader("Source Note Text")
        st.write(note_data['transcription'])
        
        st.subheader("Generated Outputs")
        col_a, col_b = st.columns(2)
        
        ext_row = summaries_df[summaries_df['method'].str.contains('Extractive')]
        abs_row = summaries_df[summaries_df['method'].str.contains('Abstractive')]
        
        if not ext_row.empty:
            col_a.markdown("**Extractive Baseline (TextRank)**")
            col_a.info(ext_row.iloc[0]['summary'])
            
        if not abs_row.empty:
            col_b.markdown("**Abstractive Model (DistilBART)**")
            col_b.success(abs_row.iloc[0]['summary'])
            
            omitted = verify_summary_faithfulness(asserted_ents, abs_row.iloc[0]['summary'])
            if omitted:
                col_b.warning(f"**Faithfulness Alert:** Active entities missing in summary: {', '.join(omitted)}")
            else:
                col_b.success("Summary retains key active clinical entities.")
    conn.close()

elif page == "Analytics":
    st.title("Specialty Entity Analytics")
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT n.medical_specialty, e.canonical_name, COUNT(*) AS freq
        FROM entities e JOIN notes n ON n.note_id = e.note_id
        WHERE e.negated = 0
        GROUP BY n.medical_specialty, e.canonical_name
        ORDER BY freq DESC
    """
    top_conditions = pd.read_sql_query(query, conn)
    st.dataframe(top_conditions)
    
    if not top_conditions.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=top_conditions, x="freq", y="canonical_name", hue="medical_specialty", ax=ax)
        st.pyplot(fig)
    conn.close()

elif page == "Search":
    st.title("Cohort Search & Concept Lookup")
    term = st.text_input("Enter condition name, drug, or CUI:")
    
    if term:
        conn = sqlite3.connect(DB_NAME)
        query = """
            SELECT DISTINCT n.note_id, n.medical_specialty, n.sample_name, n.transcription
            FROM notes n JOIN entities e ON n.note_id = e.note_id
            WHERE e.canonical_name LIKE ? OR e.text LIKE ? OR e.cui LIKE ?
        """
        results = pd.read_sql_query(query, conn, params=(f"%{term}%", f"%{term}%", f"%{term}%"))
        st.write(f"Matches found: **{len(results)}**")
        st.dataframe(results)
        conn.close()

elif page == "About / Ethics":
    st.title("Ethics, Privacy & Regulatory Compliance")
    st.markdown("""
    * **Educational Use Only:** Analytics dashboard created for capstone evaluation. Not intended for diagnostic use.
    * **Data Privacy:** Uses public de-identified transcriptions. Production deployments require anonymization pipelines complying with HIPAA, GDPR, or DPDP Act.
    * **Model Limitations:** Abstractive models carry hallucination risks; verification checks compare summaries against extracted assertions.
    """)