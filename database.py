import mysql.connector
from config import DB_CONFIG, ANIMAL_EMOJI

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def ensure_database_exists():
    # Create database if it doesn't exist.
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS animal_sounds CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error creating database: {e}")
        return False

def get_db_stats():
    try:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM audio_features")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT label, COUNT(*) as cnt FROM audio_features GROUP BY label ORDER BY cnt DESC")
        by_label = [{"label": row[0], "count": row[1], "emoji": ANIMAL_EMOJI.get(row[0])} 
                    for row in cursor.fetchall()]
        cursor.close()
        db.close()
        return {"total": total, "by_label": by_label}
    except Exception as e:
        return {"total": 0, "by_label": [], "error": str(e)}
