import sqlite3
import os
import datetime
import pandas as pd

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions.db")

def init_db():
    """Initialise the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            module TEXT,
            input_text TEXT,
            model_used TEXT,
            prediction TEXT,
            confidence REAL,
            feedback INTEGER DEFAULT NULL
        )
    ''')
    try:
        c.execute("ALTER TABLE logs ADD COLUMN feedback INTEGER DEFAULT NULL")
    except sqlite3.OperationalError:
        pass # Column already exists
    conn.commit()
    conn.close()

def log_prediction(module: str, input_text: str, model_used: str, prediction: str, confidence: float) -> int:
    """Insert a prediction record into the database and return the row ID."""
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO logs (timestamp, module, input_text, model_used, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (now, module, input_text, model_used, prediction, confidence))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def log_feedback(log_id: int, is_correct: bool):
    """Log human feedback for a specific prediction."""
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE logs SET feedback = ? WHERE id = ?", (1 if is_correct else 0, log_id))
    conn.commit()
    conn.close()

def get_all_logs() -> pd.DataFrame:
    """Retrieve all predictions as a pandas DataFrame."""
    conn = sqlite3.connect(_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_logs():
    """Clear all records from the logs table."""
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
