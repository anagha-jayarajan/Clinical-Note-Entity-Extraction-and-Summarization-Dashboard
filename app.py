import streamlit as st
import pandas as pd
from src.preprocess import load_and_clean_data
from src.extract import load_nlp_pipeline, extract_entities
from src.summarize import get_abstractive_summarizer, summarize_extractive, summarize_abstractive
from src.db import init_db, populate_db, get_connection

# Page Configuration
st.set_page_config(
    page_title="Clinical NLP Dashboard",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_all_models():
    """Loads and caches heavy NLP models in memory to prevent cold-start reloads."""
    nlp = load_nlp_pipeline()
    abs_summarizer = get_abstractive_summarizer()
    return nlp, abs_summarizer

@st.cache_data
def load_and_prepare_data():
    """Loads and cleans the dataset once."""
    return load_and_clean_data()

def main():
    st.title("🏥 Clinical NLP Analytics & Summarization Dashboard")
    st.markdown("Automated Named Entity Recognition (NER), Negation Detection, and Dual-Method Summarization.")

    # Display spinner during initial resource load
    with st.spinner("Loading clinical NLP models and initializing database... Please wait."):
        nlp, abs_summarizer = load_all_models()
        init_db()
        df = load_and_prepare_data()
        
        # Populate database if empty
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            populate_db(df.head(50), nlp, abs_summarizer)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Module:", [
        "Overview & Analytics",
        "Note Explorer & NER",
        "Summarization Comparison",
        "Cohort Search"
    ])

    if page == "Overview & Analytics":
        st.header("Pipeline Analytics Overview")
        st.write(f"Total Processed Notes: **{len(df)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Specialties")
            st.bar_chart(df['medical_specialty'].value_counts().head(10))
            
        with col2:
            st.subheader("Dataset Preview")
            st.dataframe(df[['note_id', 'medical_specialty', 'description']].head(10))

    elif page == "Note Explorer & NER":
        st.header("Interactive Clinical NER & Negation Detector")
        selected_id = st.selectbox("Select Note ID:", df['note_id'].tolist())
        
        note_data = df[df['note_id'] == selected_id].iloc[0]
        st.text_area("Original Clinical Transcription:", note_data['transcription'], height=250)
        
        if st.button("Run Entity Extraction"):
            entities = extract_entities(nlp, note_data['transcription'])
            if entities:
                ent_df = pd.DataFrame(entities)
                st.subheader("Extracted Medical Entities")
                st.dataframe(ent_df)
            else:
                st.info("No clinical entities identified in this note.")

    elif page == "Summarization Comparison":
        st.header("Extractive vs. Abstractive Summarization")
        selected_id = st.selectbox("Select Note ID for Summarization:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        st.subheader("Source Note Description")
        st.info(note_data['description'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Extractive Summary (TextRank)")
            ext_summary = summarize_extractive(note_data['transcription'])
            st.write(ext_summary)
            
        with col2:
            st.subheader("Abstractive Summary (DistilBART)")
            abs_summary = summarize_abstractive(abs_summarizer, note_data['transcription'])
            st.write(abs_summary)

    elif page == "Cohort Search":
        st.header("Clinical Cohort Search Engine")
        search_term = st.text_input("Enter condition, drug, or clinical keyword:", "hypertension")
        
        if search_term:
            filtered_df = df[df['transcription'].str.contains(search_term, case=False, na=False)]
            st.write(f"Found **{len(filtered_df)}** matching notes for term: *{search_term}*")
            st.dataframe(filtered_df[['note_id', 'medical_specialty', 'description']])

if __name__ == "__main__":
    main()
