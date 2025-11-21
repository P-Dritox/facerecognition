import os
import json
from mysql.connector import pooling

_DBPOOL = None

def _init_dbpool():
    global _DBPOOL
    if _DBPOOL is not None:
        return
    ca_path = "/etc/secrets/server-ca.pem"
    if not os.path.exists(ca_path):
        print(f"Warning: CA file not found at {ca_path}")
        return
    _DBPOOL = pooling.MySQLConnectionPool(
        pool_name="kaizen_pool",
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        pool_reset_session=True,
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME", "bdKaizen"),
        port=int(os.getenv("DB_PORT", 3306)),
        ssl_ca=ca_path,
        ssl_verify_cert=True,
        connection_timeout=15,
        charset="utf8mb4"
    )

def get_db_connection():
    _init_dbpool()
    if _DBPOOL is None:
        return None
    try:
        return _DBPOOL.get_connection()
    except Exception as e:
        print(f"Error getting DB connection: {e}")
        return None

def get_all_staff_embeddings():
    """
    Obtiene todos los embeddings de staff desde la base de datos
    Retorna una lista de diccionarios con: {id, name, embedding}
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT StaffID, CONCAT(StFirstName, ' ', StLastName) as name, FaceEmbedding 
            FROM rhStaff 
            WHERE FaceEmbedding IS NOT NULL
        """)
        rows = cur.fetchall()
        
        staff_list = []
        for row in rows:
            try:
                embedding = json.loads(row['FaceEmbedding']) if isinstance(row['FaceEmbedding'], str) else row['FaceEmbedding']
                if embedding and isinstance(embedding, list):
                    staff_list.append({
                        'id': row['StaffID'],
                        'name': row['name'],
                        'embedding': embedding
                    })
            except Exception as e:
                print(f"Error parsing embedding for staff {row['StaffID']}: {e}")
                continue
        
        return staff_list
    except Exception as e:
        print(f"Error fetching staff embeddings: {e}")
        return []
    finally:
        try:
            cur.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass