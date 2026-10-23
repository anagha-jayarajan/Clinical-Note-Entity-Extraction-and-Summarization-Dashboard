import sqlite3
import pandas as pd

DB_NAME = "clinical_nlp.db"

def init_db(db_path: str = DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS summaries")
    cursor.execute("DROP TABLE IF EXISTS entities")
    cursor.execute("DROP TABLE IF EXISTS notes")

    cursor.execute("""
        CREATE TABLE notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            medical_specialty TEXT,
            sample_name TEXT,
            transcription TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            text TEXT,
            label TEXT,
            negated INTEGER,
            cui TEXT,
            canonical_name TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (note_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            method TEXT,
            summary TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (note_id)
        )
    """)
    conn.commit()
    conn.close()

def query_top_conditions(db_path: str = DB_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT n.medical_specialty, e.canonical_name, COUNT(*) AS frequency
        FROM entities e
        JOIN notes n ON n.note_id = e.note_id
        WHERE e.negated = 0
        GROUP BY n.medical_specialty, e.canonical_name
        ORDER BY frequency DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df