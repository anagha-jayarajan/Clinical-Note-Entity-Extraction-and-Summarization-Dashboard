import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_and_clean_data(file_path: str = "data/mtsamples.csv") -> pd.DataFrame:
    """
    Loads, cleans, and standardizes the clinical transcription dataset.
    Ensures that essential columns exist and missing records are safely handled.
    """
    # Fallback path check
    if not os.path.exists(file_path):
        # Check alternative common locations
        alt_paths = ["mtsamples.csv", "../data/mtsamples.csv", "src/data/mtsamples.csv"]
        for alt in alt_paths:
            if os.path.exists(alt):
                file_path = alt
                break

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback dummy dataset if file is missing to keep container running
        df = pd.DataFrame({
            'note_id': [1, 2],
            'medical_specialty': ['Cardiology', 'Neurology'],
            'description': ['Sample clinical note description.', 'Sample neurological evaluation.'],
            'transcription': [
                'Patient presents with severe hypertension and shortness of breath. No chest pain.',
                'Patient reports no history of severe headaches or seizures. Normal EEG findings.'
            ]
        })
        return df

    # Standardize column names (lowercase and strip whitespace)
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Clean missing critical fields
    if 'transcription' in df.columns:
        df = df.dropna(subset=['transcription'])
        df = df[df['transcription'].str.strip() != '']
    else:
        df['transcription'] = "No transcription provided."

    # Fill default values for optional columns
    if 'medical_specialty' in df.columns:
        df['medical_specialty'] = df['medical_specialty'].fillna('Unassigned / Unknown').str.strip()
    else:
        df['medical_specialty'] = 'Unassigned / Unknown'

    if 'description' in df.columns:
        df['description'] = df['description'].fillna('No description provided.').str.strip()
    else:
        df['description'] = 'No description provided.'

    # Ensure 'note_id' column exists explicitly
    if 'note_id' not in df.columns:
        if 'id' in df.columns:
            df['note_id'] = df['id']
        elif 'unnamed: 0' in df.columns:
            df['note_id'] = df['unnamed: 0']
        else:
            df = df.reset_index(drop=True)
            df['note_id'] = df.index + 1

    # Ensure note_id is numeric
    df['note_id'] = pd.to_numeric(df['note_id'], errors='coerce').fillna(0).astype(int)

    return df
