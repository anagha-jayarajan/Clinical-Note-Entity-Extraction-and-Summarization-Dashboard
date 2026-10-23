import streamlit as st
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Page Configuration (Must be first Streamlit call)
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

# 2. Main Entry Point with Exception Handling
def main():
    st.markdown("""
        <div class="header-box">
            <h1>🏥 Clinical NLP Intelligence Studio</h1>
            <p>Automated Named Entity Recognition (NER), Negation Detection & Benchmarked Summarization</p>
        </div>
    """, unsafe_allow_html=True)

    # Wrap imports to prevent container crash
    try:
        with st.spinner("Loading NLP components and database..."):
            from src.preprocess import load_and_clean_data
            from src.extract import load_nlp_pipeline, extract_entities
            from src.summarize import get_abstractive_summarizer, summarize_extractive, summarize_abstractive
            from src.db import init_db, populate_db, get_connection
            
            init_db()
            df = load_and_clean_data()
            
            # Check DB count
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM notes")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0:
                populate_db(df.head(20))
                
    except Exception as e:
        st.error("⚠️ An error occurred while initializing application modules:")
        st.exception(e)
        st.info("Please fix the file import error indicated above.")
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
            st.metric("Total Records", len(df))
            st.bar_chart(df['medical_specialty'].value_counts().head(8))
        with col2:
            st.dataframe(df[['note_id', 'medical_specialty', 'description']].head(10), use_container_width=True)

    elif page == "🔬 Clinical Entity Extraction":
        st.subheader("Named Entity Recognition")
        nlp = load_nlp_pipeline()
        selected_id = st.selectbox("Select Note ID:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        st.text_area("Clinical Note", note_data['transcription'], height=250)
        if st.button("Extract Entities", type="primary"):
            entities = extract_entities(nlp, note_data['transcription'])
            if entities:
                st.dataframe(entities, use_container_width=True)
            else:
                st.info("No entities found.")

    elif page == "📝 Summarization Studio":
        st.subheader("Summarization")
        abs_summarizer = get_abstractive_summarizer()
        selected_id = st.selectbox("Select Note ID:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Extractive Summary**")
            st.write(summarize_extractive(note_data['transcription']))
        with c2:
            st.markdown("**Abstractive Summary**")
            st.write(summarize_abstractive(abs_summarizer, note_data['transcription']))

    elif page == "🔍 Cohort Search Engine":
        st.subheader("Cohort Search")
        query = st.text_input("Enter condition/keyword:", "hypertension")
        if query:
            filtered = df[df['transcription'].str.contains(query, case=False, na=False)]
            st.write(f"Found **{len(filtered)}** matches.")
            st.dataframe(filtered[['note_id', 'medical_specialty', 'description']], use_container_width=True)

if __name__ == "__main__":
    main()
