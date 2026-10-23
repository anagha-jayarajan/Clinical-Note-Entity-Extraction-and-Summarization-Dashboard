import sqlite3
import pandas as pd

DB_NAME = "clinical_nlp.db"

def get_connection():
    """Returns a connection object to the SQLite database."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for raw notes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            note_id INTEGER PRIMARY KEY,
            medical_specialty TEXT,
            transcription TEXT,
            description TEXT
        )
    """)
    
    # Table for extracted entities
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            entity_text TEXT,
            entity_label TEXT,
            is_negated BOOLEAN,
            FOREIGN KEY (note_id) REFERENCES notes (note_id)
        )
    """)
    
    # Table for summaries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            note_id INTEGER PRIMARY KEY,
            extractive_summary TEXT,
            abstractive_summary TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (note_id)
        )
    """)
    
    conn.commit()
    conn.close()

def populate_db(df: pd.DataFrame, nlp, abs_summarizer):
    """Populates the SQLite database with notes, extracted entities, and summaries."""
    conn = get_connection()
    cursor = conn.cursor()
    
    for _, row in df.iterrows():
        note_id = int(row['note_id'])
        specialty = str(row['medical_specialty'])
        text = str(row['transcription'])
        desc = str(row['description'])
        
        # Insert note
        cursor.execute(
            "INSERT OR REPLACE INTO notes (note_id, medical_specialty, transcription, description) VALUES (?, ?, ?, ?)",
            (note_id, specialty, text, desc)
        )
        
        # Extract and insert entities
        doc = nlp(text)
        for ent in doc.ents:
            is_negated = getattr(ent._, "negex", False)
            cursor.execute(
                "INSERT INTO entities (note_id, entity_text, entity_label, is_negated) VALUES (?, ?, ?, ?)",
                (note_id, ent.text, ent.label_, is_negated)
            )
            
    conn.commit()
    conn.close()
