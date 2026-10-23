# Clinical-Note-Entity-Extraction-and-Summarization-Dashboard

Live demo : https://clinical-note-entity-extraction-and-summarization-dashboard-5q.streamlit.app/

## Project Overview

The **Clinical Natural Language Processing Analytics and Summarization Dashboard** is an end-to-end biomedical Natural Language Processing (NLP) application designed to extract clinically relevant information, identify negated medical findings, and generate concise summaries from clinical transcriptions. The project uses the MTSamples dataset and integrates biomedical NLP models, text summarization techniques, quantitative evaluation methods, and a SQLite analytical database into an interactive Streamlit dashboard.

The system supports two summarization approaches—extractive and abstractive summarization—and provides functionality for exploring medical entities, comparing generated summaries, analyzing clinical specialties, and searching clinical note cohorts.

This project demonstrates the application of NLP and machine learning techniques to clinical text analytics, with an emphasis on information extraction, summarization, evaluation, and responsible handling of medical text.

## Key Features

### 1. Biomedical Named Entity Recognition

The application uses the scispaCy biomedical NLP model `en_ner_bc5cdr_md` to identify clinically relevant entities in unstructured clinical text.

Key capabilities include:
- Extraction of disease and chemical entities.
- Identification of medical conditions and medication-related mentions supported by the selected model.
- Structured representation of extracted entities for downstream analysis.
- Comparison of predicted entities against manually annotated reference data.

### 2. Clinical Negation Detection

The pipeline integrates negspaCy with the NegEx approach to identify negated clinical findings.

For example, the system distinguishes between:

- **Positive assertion:** "The patient has shortness of breath."
- **Negated assertion:** "The patient denies shortness of breath."

This distinction helps prevent explicitly negated findings from being treated as active clinical conditions during analysis.

### 3. Dual-Method Text Summarization

The application implements two complementary summarization methods.

**Extractive Summarization — TextRank**

- Uses a graph-based ranking algorithm through the Sumy framework.
- Identifies important sentences from the original clinical transcription.
- Produces summaries using sentences selected from the source text.
- Provides an interpretable baseline for summarization evaluation.

**Abstractive Summarization — DistilBART**

- Uses a sequence-to-sequence transformer model through Hugging Face Transformers.
- Generates a concise summary based on the input transcription.
- Can produce rephrased sentences rather than only copying sentences from the source.
- Enables comparison with the extractive baseline.

### 4. Clinical Faithfulness Verification

The project includes an automated verification step intended to check whether active medical entities identified in the source note are retained in the generated abstractive summary.

This check supports the assessment of clinical information preservation. However, entity retention alone does not establish complete clinical correctness, and generated summaries require further validation before any real-world clinical use.

### 5. SQLite Analytical Database

SQLite provides structured local storage for clinical transcription data and analytical outputs.

Depending on the implemented database schema, stored information can include:
- Source clinical notes and cleaned text.
- Extracted medical entities.
- Concept Unique Identifiers (CUIs), where available.
- Negation and assertion flags.
- Extractive and abstractive summaries.
- Evaluation outputs and other analysis results.

The database supports structured queries and enables the dashboard to retrieve information for exploration and comparison.

### 6. Interactive Streamlit Dashboard

The Streamlit interface organizes the system into analytical modules:

- **Analytics Overview:** Displays overall pipeline statistics and key metrics.
- **Note Explorer:** Supports inspection of individual clinical notes and extracted information.
- **Summary Comparison:** Compares extractive and abstractive summaries.
- **Specialty Analytics:** Explores clinical notes by medical specialty.
- **Cohort Search:** Supports searching and filtering groups of clinical notes.
- **Ethics and Privacy:** Presents the project's intended use, data considerations, and limitations.

The availability of each module depends on the corresponding implementation in the application.

## System Architecture

The system follows a modular workflow that transforms clinical transcriptions into structured analytical outputs.

1. **Data ingestion:** Load clinical transcriptions from the MTSamples dataset.
2. **Preprocessing:** Clean and normalize the text for downstream NLP tasks.
3. **Entity extraction:** Apply the scispaCy biomedical NLP pipeline.
4. **Negation detection:** Identify negated clinical findings using negspaCy.
5. **Text summarization:** Generate extractive summaries with TextRank and abstractive summaries with DistilBART.
6. **Faithfulness verification:** Compare active source-note entities with entities retained in the generated summaries.
7. **Database storage:** Store structured notes, entities, flags, summaries, and supported analytical results in SQLite.
8. **Quantitative evaluation:** Evaluate entity extraction and summarization using reference annotations and ROUGE metrics.
9. **Dashboard visualization:** Present the results through the Streamlit interface.

### Architecture Diagram

```text
                 MTSamples Dataset
                         |
                         v
                Data Preprocessing
                         |
                         v
             Clinical Text Processing
                         |
              +----------+----------+
              |                     |
              v                     v
       Biomedical NER         Negation Detection
         (scispaCy)             (negspaCy)
              |                     |
              +----------+----------+
                         |
                         v
                Text Summarization
                         |
              +----------+----------+
              |                     |
              v                     v
         TextRank               DistilBART
       (Extractive)             (Abstractive)
              |                     |
              +----------+----------+
                         |
                         v
             Clinical Faithfulness Check
                         |
                         v
                SQLite Database
                         |
                         v
                Streamlit Dashboard
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Analytics      Note Explorer   Summary Comparison
          |
          v
   Specialty Analytics, Cohort Search,
      and Ethics/Privacy Information
```

## Project Directory Structure

The following is the intended project organization. Adjust filenames and directories to match the actual repository contents.

```text
clinical-nlp-dashboard/
│
├── data/
│   ├── annotations/
│   │   └── gold_sample.json
│   │
│   └── raw/
│       └── mtsamples.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── extract.py
│   ├── summarize.py
│   ├── db.py
│   └── evaluate.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

### Directory and File Descriptions

| File or Directory | Purpose |
|---|---|
| `data/raw/` | Contains the source clinical transcription dataset. |
| `data/annotations/` | Contains manually annotated reference data for evaluation. |
| `src/preprocess.py` | Handles data loading, cleaning, and text normalization. |
| `src/extract.py` | Implements biomedical entity extraction and negation detection. |
| `src/summarize.py` | Implements extractive and abstractive summarization. |
| `src/db.py` | Defines SQLite database operations and analytical queries. |
| `src/evaluate.py` | Implements supported evaluation procedures and metric calculations. |
| `app.py` | Provides the Streamlit dashboard entry point. |
| `requirements.txt` | Lists Python dependencies required by the application. |
| `.gitignore` | Excludes generated files and other items that should not be committed. |
| `README.md` | Documents the project, setup, architecture, and evaluation. |

The annotation file, dataset, and source modules listed above must be present in the repository for the corresponding functionality to run. Do not commit restricted, sensitive, or unnecessarily large data files.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard and Visualization | Streamlit, Matplotlib, Seaborn |
| Biomedical NLP | spaCy, scispaCy, negspaCy |
| Biomedical NER Model | `en_ner_bc5cdr_md` |
| Extractive Summarization | TextRank, Sumy |
| Abstractive Summarization | DistilBART, Hugging Face Transformers |
| Deep Learning Framework | PyTorch |
| Text Processing | NLTK |
| Evaluation | ROUGE-Score, precision, recall, F1-score |
| Database | SQLite |
| Data Handling | pandas and Python standard libraries, where used |

## Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or another Python version compatible with all selected dependencies.
- Git.
- pip, the Python package installer.
- An internet connection for initial dependency and model downloads.
- Sufficient RAM and disk space to load transformer models and process the dataset.

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with the GitHub account that hosts the repository.

```bash
git clone https://github.com/YOUR_USERNAME/clinical-nlp-dashboard.git
cd clinical-nlp-dashboard
```

### Step 2: Create a Virtual Environment

Create an isolated Python environment to manage the project's dependencies.

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

If PowerShell prevents activation because of its execution policy, use an approved environment-specific workaround rather than changing system-wide security settings unnecessarily.

### Step 3: Install Dependencies

Upgrade the package installation tools and install the dependencies listed in the repository.

```bash
python -m pip install --upgrade pip "setuptools<82" wheel
pip install -r requirements.txt
```

The `requirements.txt` file should contain compatible versions of all required packages, including Streamlit, spaCy, scispaCy, negspaCy, Sumy, Transformers, PyTorch, NLTK, ROUGE-Score, pandas, Matplotlib, and Seaborn, as applicable to the implementation.

### Step 4: Install the Biomedical NLP Model

If the biomedical model is not installed automatically by the project setup, install the version compatible with the selected scispaCy release.

For example, for a compatible scispaCy 0.5.3 environment:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
```

If you use another scispaCy release, select the matching model release rather than mixing incompatible versions.

### Step 5: Download Required NLTK Resources

Download the tokenizer resources required by the project's sentence-processing implementation.

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The exact resources required depend on the installed NLTK version and the functions used by the application.

### Step 6: Verify the Dataset and Project Files

Before launching the dashboard, confirm that:

- The dataset is available at the path expected by the application.
- The source modules exist in the `src/` directory.
- The annotation file is available if evaluation is required.
- The configured database path is writable.
- All required environment variables or configuration values are set, if applicable.

### Step 7: Launch the Application

From the root directory, run:

```bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal. The default address is:

`http://localhost:8501`

Open this address in a browser if it does not open automatically.

## Evaluation Framework

The project evaluates two main tasks: biomedical named entity recognition and clinical text summarization.

The reported results below represent the project's stated evaluation figures. They should be treated as reported results rather than independently verified benchmarks; reproducibility depends on the underlying annotations, evaluation scripts, model versions, and experimental configuration.

### 1. Named Entity Recognition Evaluation

Named Entity Recognition (NER) performance is evaluated using precision, recall, and F1-score against the manually annotated reference dataset.

| Metric | Reported Score |
|---|---:|
| Precision | 0.8571 |
| Recall | 0.8571 |
| F1-score | 0.8571 |

**Interpretation**

- **Precision:** The proportion of predicted entities that match the reference annotations.
- **Recall:** The proportion of reference entities successfully identified by the model.
- **F1-score:** The harmonic mean of precision and recall.

The reported scores indicate that the system identifies a substantial proportion of the annotated entities while maintaining a similar balance between precision and recall.

The evaluation should specify the entity types, exact-match criteria, annotation conventions, test-set size, and handling of duplicate predictions to make the results reproducible.

### 2. Summarization Evaluation

The extractive and abstractive summarization approaches are compared using ROUGE metrics against reference summaries.

| Summarization Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| TextRank — Extractive Baseline | 0.3241 | 0.1425 | 0.2810 |
| DistilBART — Abstractive Pipeline | 0.4120 | 0.2105 | 0.3650 |

**Interpretation**

- **ROUGE-1:** Measures unigram overlap between a generated summary and its reference.
- **ROUGE-2:** Measures bigram overlap.
- **ROUGE-L:** Measures overlap based on the longest common subsequence.

The reported results show higher ROUGE scores for DistilBART than for TextRank across all three metrics. This suggests that the abstractive pipeline achieves better lexical overlap with the selected reference summaries on the evaluated dataset.

However, ROUGE scores alone do not establish clinical correctness, factual consistency, completeness, or patient safety. Clinical summarization should also be assessed for omitted findings, unsupported statements, negation errors, and preservation of relevant medical information.

### 3. Running the Evaluation Suite

If the evaluation module and its required data are available, run the evaluation from the project root:

```bash
python -m src.evaluate
```

The script should load the reference annotations, execute the relevant evaluation procedures, and report the metrics implemented in the module.

For reproducibility, record the dataset version, annotation methodology, train/test split, model configuration, dependency versions, and evaluation parameters alongside the results.

## Privacy, Ethics, and Responsible Use

### Data Source

The project is designed around the publicly available MTSamples clinical transcription corpus. Public availability does not, by itself, guarantee that every record is free of identifying information or that every intended use is authorized. Verify the dataset's terms, provenance, and privacy status before redistribution or deployment.

### Clinical Safety

This application is intended for academic learning, research, and demonstration purposes. It is not a medical device and must not be used as a standalone system for diagnosis, treatment selection, or other clinical decision-making.

Generated summaries and extracted entities may contain errors. In particular, incorrect negation detection, missing entities, and unsupported abstractive statements can alter the meaning of a clinical note.

### Privacy and Data Protection

Any deployment involving identifiable health information must use appropriate access controls, data minimization, secure storage, retention policies, and applicable legal and institutional safeguards.

Depending on the jurisdiction and use case, relevant requirements may include HIPAA, GDPR, and applicable Indian data protection laws. Compliance cannot be established by anonymization alone; it requires an assessment of the full data-processing and deployment workflow.

### Model Limitations

- Biomedical NER performance depends on the model's training data and supported entity categories.
- Negation detection may not correctly interpret every clinical phrase or contextual assertion.
- Abstractive models may omit important details or generate unsupported information.
- ROUGE scores measure textual overlap rather than clinical validity.
- Entity-based faithfulness checks do not guarantee that the full meaning of a source note is preserved.

Any extension toward real clinical use would require appropriate expert review, validation on representative data, privacy assessment, and a clearly defined safety process.

## Reproducibility and Future Improvements

Potential directions for extending the project include:

1. Expanding the manually annotated evaluation dataset across clinical specialties.
2. Reporting per-entity precision, recall, and F1-score in addition to aggregate metrics.
3. Evaluating negation detection separately against annotated clinical assertions.
4. Adding clinical summarization assessments for factual consistency, completeness, and information retention.
5. Comparing additional summarization models under a consistent evaluation protocol.
6. Improving cohort search with configurable filters and structured entity queries.
7. Adding automated tests for preprocessing, entity extraction, summarization, and database operations.
8. Pinning dependency versions and documenting hardware requirements for reproducible deployment.
9. Improving error handling, logging, and dashboard performance for larger datasets.
10. Documenting privacy controls and dataset provenance more explicitly.

## Conclusion

The Clinical Natural Language Processing Analytics and Summarization Dashboard combines biomedical entity extraction, negation detection, dual-method text summarization, automated entity-retention checks, SQLite storage, and interactive analytics in a modular Python application.

By integrating these components into a single workflow, the project provides a practical framework for exploring how NLP and transformer-based models can support the organization and summarization of unstructured clinical text. Its evaluation framework and dashboard also provide a foundation for further experiments in biomedical information extraction, clinical text analytics, and responsible NLP system development.

## License and Attribution

Add a license file before distributing the project if you intend to make the source code available for reuse. Ensure that the MTSamples dataset and all third-party models, libraries, and resources are used in accordance with their respective licenses and terms.

## Author and Contributions

**Project:** Clinical Natural Language Processing Analytics and Summarization Dashboard

**Domain:** Biomedical NLP, Clinical Text Analytics, Machine Learning, and Text Summarization

**Implementation:** Python, Streamlit, scispaCy, negspaCy, Transformers, and SQLite

For questions, collaboration, or academic discussion, contact the repository maintainer through the contact details provided in the GitHub repository.

