import streamlit as st
import warnings
import gc

# Suppress non-fatal warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Page Configuration
st.set_page_config(
    page_title="Clinical NLP Studio",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .header-box {
        background: linear-gradient(135deg, #0e1e38 0%, #1a365d 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .header-box h1 { color: #ffffff !important; margin-bottom: 4px !important; }
    .header-box p { color: #cbd5e1 !important; margin: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Resource Caching Loaders
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    from src.extract import load_nlp_pipeline
    return load_nlp_pipeline()

@st.cache_resource(show_spinner=False)
def load_transformer_summarizer():
    from src.summarize import get_abstractive_summarizer
    return get_abstractive_summarizer()

@st.cache_data(show_spinner=False)
def get_clean_dataset():
    from src.preprocess import load_and_clean_data
    return load_and_clean_data()

def init_database_if_needed(df):
    from src.db import init_db, populate_db, get_connection
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM notes")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        populate_db(df)

def main():
    st.markdown("""
        <div class="header-box">
            <h1>🏥 Clinical NLP Intelligence Studio</h1>
            <p>Automated Named Entity Recognition (NER), Negation Detection & Benchmarked Summarization</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Initializing system components & dataset..."):
            df = get_clean_dataset()
            init_database_if_needed(df)
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return

    # Navigation Sidebar
    st.sidebar.title("Clinical NLP Suite")
    page = st.sidebar.radio("Navigation", [
        "📊 Executive Overview",
        "🔬 Clinical Entity Extraction",
        "📝 Summarization Studio",
        "🔍 Cohort Search Engine"
    ])

    if page == "📊 Executive Overview":
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records Loaded", len(df))
            st.bar_chart(df['medical_specialty'].value_counts().head(8))
        with col2:
            st.dataframe(df[['note_id', 'medical_specialty', 'description']].head(10), width="stretch")

    elif page == "🔬 Clinical Entity Extraction":
        st.subheader("Named Entity Recognition")
        nlp = load_spacy_model()
        from src.extract import extract_entities
        
        selected_id = st.selectbox("Select Note ID:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        st.text_area("Clinical Note", note_data['transcription'], height=250)
        if st.button("Extract Entities", type="primary"):
            entities = extract_entities(nlp, note_data['transcription'])
            if entities:
                st.dataframe(entities, width="stretch")
            else:
                st.info("No entities found.")
            gc.collect()

    elif page == "📝 Summarization Studio":
        st.subheader("Summarization Engine")
        abs_summarizer = load_transformer_summarizer()
        from src.summarize import summarize_extractive, summarize_abstractive
        
        selected_id = st.selectbox("Select Note ID:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Extractive Summary (TextRank)**")
            st.write(summarize_extractive(note_data['transcription']))
        with c2:
            st.markdown("**Abstractive Summary (DistilBART)**")
            st.write(summarize_abstractive(abs_summarizer, note_data['transcription']))
        gc.collect()

    elif page == "🔍 Cohort Search Engine":
        st.subheader("Cohort Search")
        query = st.text_input("Enter condition/keyword:", "hypertension")
        if query:
            filtered = df[df['transcription'].str.contains(query, case=False, na=False)]
            st.write(f"Found **{len(filtered)}** matches.")
            st.dataframe(filtered[['note_id', 'medical_specialty', 'description']], width="stretch")

if __name__ == "__main__":
    main()
