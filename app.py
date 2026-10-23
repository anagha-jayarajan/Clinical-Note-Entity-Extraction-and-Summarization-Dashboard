import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Clinical NLP Studio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Enterprise/Clinical UI Styling
st.markdown("""
    <style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #0e1e38 0%, #1a365d 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }
    .header-box p {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: left;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Subheader Styling */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Cached Resource Loaders
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
        populate_db(df.head(50))

def main():
    # Application Top Banner
    st.markdown("""
        <div class="header-box">
            <h1>🏥 Clinical NLP Intelligence Studio</h1>
            <p>Automated Named Entity Recognition (NER), Negation Detection & Benchmarked Clinical Summarization</p>
        </div>
    """, unsafe_allow_html=True)

    # Initial Loading Routine
    with st.spinner("Initializing models and analytical database..."):
        try:
            df = get_clean_dataset()
            nlp = load_spacy_model()
            abs_summarizer = load_transformer_summarizer()
            init_database_if_needed(df)
        except Exception as e:
            st.error(f"Initialization error: {e}")
            return

    # Sidebar Options
    st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.sidebar.title("Clinical NLP Suite")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation Menu", [
        "📊 Executive Overview",
        "🔬 Clinical Entity Extraction",
        "📝 Summarization Studio",
        "🔍 Cohort Search Engine"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: **Active** | Model: **scispaCy + DistilBART**")

    # View 1: Executive Overview
    if page == "📊 Executive Overview":
        st.markdown('<div class="section-title">Dataset & Pipeline Analytics</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""<div class="metric-card"><div class="label">Total Clinical Notes</div><div class="value">{}</div></div>""".format(len(df)), unsafe_allow_html=True)
        with m2:
            st.markdown("""<div class="metric-card"><div class="label">Medical Specialties</div><div class="value">{}</div></div>""".format(df['medical_specialty'].nunique()), unsafe_allow_html=True)
        with m3:
            st.markdown("""<div class="metric-card"><div class="label">NER Model</div><div class="value" style="font-size:1.2rem; padding-top:8px;">en_ner_bc5cdr_md</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown("""<div class="metric-card"><div class="label">Summarizer</div><div class="value" style="font-size:1.2rem; padding-top:8px;">DistilBART-CNN</div></div>""", unsafe_allow_html=True)

        st.write("")
        st.write("")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Specialty Distribution")
            st.bar_chart(df['medical_specialty'].value_counts().head(8))
            
        with col2:
            st.subheader("Data Explorer Preview")
            st.dataframe(
                df[['note_id', 'medical_specialty', 'description']].head(10),
                use_container_width=True,
                height=300
            )

    # View 2: Clinical Entity Extraction
    elif page == "🔬 Clinical Entity Extraction":
        st.markdown('<div class="section-title">Named Entity Recognition (NER) & Negation Context</div>', unsafe_allow_html=True)
        
        from src.extract import extract_entities
        
        col_sel, col_empty = st.columns([1, 2])
        with col_sel:
            selected_id = st.selectbox("Select Target Note ID:", df['note_id'].tolist())
            
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        tab1, tab2 = st.tabs(["📄 Full Transcription", "🏷 Extracted Entities"])
        
        with tab1:
            st.text_area("Clinical Note Context", note_data['transcription'], height=320)
            
        with tab2:
            if st.button("⚡ Extract Medical Entities", type="primary"):
                with st.spinner("Analyzing text for diseases, chemicals, and negation status..."):
                    entities = extract_entities(nlp, note_data['transcription'])
                    if entities:
                        ent_df = pd.DataFrame(entities)
                        st.dataframe(ent_df, use_container_width=True)
                    else:
                        st.info("No clinical entities identified in this note.")
            else:
                st.caption("Click the button above to run entity recognition on this note.")

    # View 3: Summarization Studio
    elif page == "📝 Summarization Studio":
        st.markdown('<div class="section-title">Dual-Method Clinical Note Summarization</div>', unsafe_allow_html=True)
        from src.summarize import summarize_extractive, summarize_abstractive
        
        selected_id = st.selectbox("Select Note ID to Summarize:", df['note_id'].tolist())
        note_data = df[df['note_id'] == selected_id].iloc[0]
        
        st.info(f"**Chief Complaint / Description:** {note_data['description']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Extractive (TextRank)")
            with st.container(border=True):
                ext_summary = summarize_extractive(note_data['transcription'])
                st.write(ext_summary if ext_summary else "No summary available.")
                
        with c2:
            st.subheader("Abstractive (DistilBART)")
            with st.container(border=True):
                abs_summary = summarize_abstractive(abs_summarizer, note_data['transcription'])
                st.write(abs_summary if abs_summary else "No summary available.")

    # View 4: Cohort Search Engine
    elif page == "🔍 Cohort Search Engine":
        st.markdown('<div class="section-title">Clinical Cohort & Keyword Search</div>', unsafe_allow_html=True)
        
        search_term = st.text_input("Enter condition, symptom, or medication:", "hypertension")
        
        if search_term:
            filtered_df = df[df['transcription'].str.contains(search_term, case=False, na=False)]
            st.markdown(f"Found **{len(filtered_df)}** matching clinical records for term: `'{search_term}'`")
            
            st.dataframe(
                filtered_df[['note_id', 'medical_specialty', 'description', 'transcription']],
                use_container_width=True,
                height=400
            )

if __name__ == "__main__":
    main()
