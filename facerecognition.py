import os

# ---- Reducir ruido de TensorFlow/absl ANTES de importar deepface/tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # fuerza CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # 0=ALL, 1=INFO-, 2=WARNING-, 3=ERROR-
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # menos logs oneDNN

from flask import Flask, request, Response
from flask_cors import CORS
import requests
import cv2
import numpy as np
from deepface import DeepFace
import uuid
import json
import mysql.connector
import base64
import gc

# Intentar bajar aún más la verbosidad de absl (si está disponible)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# ============================================================
# CONFIGURACIÓN APP
# ============================================================
app = Flask(__name__)
CORS(app)

VERBOSE = os.getenv("VERBOSE_LOGS", "1") == "1"  

def log(msg):
    print(msg, flush=True)

def dlog(msg):
    if VERBOSE:
        print(msg, flush=True)

# ============================================================
# PRECARGA (API pública) PARA USAR CACHÉ INTERNA
# ============================================================
log("[INIT] Precargando modelo Facenet...")
try:
    _ = DeepFace.build_model("Facenet")  
    log("[INIT] Modelo cargado correctamente ✅")
except Exception as e:
    log(f"[INIT][WARN] No se pudo precargar Facenet: {e}. Se cargará on-demand.")

# ============================================================
# DB CONEXIÓN
# ============================================================
def get_db_connection():
    """
    Conexión segura a MySQL (GCP) usando certificado SSL (solo CA).
    Variables en Render:
      DB_HOST, DB_USER, DB_PASS, DB_NAME (bdKaizen), DB_PORT (3306)
    """
    try:
        ca_path = "/etc/secrets/server-ca.pem"
        if not os.path.exists(ca_path):
            log(f"[ERROR] Certificado no encontrado en {ca_path}")
            return None

        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME", "bdKaizen"),
            port=int(os.getenv("DB_PORT", 3306)),
            ssl_ca=ca_path,
            ssl_verify_cert=True,
            autocommit=True,
            connection_timeout=10
        )
        return conn
    except mysql.connector.Error as err:
        log(f"[ERROR] Error MySQL: {err}")
        return None
    except Exception as e:
        log(f"[ERROR] Error general de conexión: {e}")
        return None

# ============================================================
# AUXILIARES
# ============================================================
def download_image_from_drive(file_id):
    """
    Descarga una imagen desde Google Drive y la devuelve como matriz OpenCV (BGR).
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        dlog(f"[INFO] Descargando imagen desde: {url}")
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            log(f"[ERROR] No se pudo descargar la imagen (HTTP {response.status_code}) con file_id: {file_id}")
            return None
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        log(f"[ERROR] Error al descargar la imagen: {e}")
        return None


def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes de rostros y devuelve la distancia.
    Lógica intacta: similarity = 1 - distance.
    """
    try:
        result = DeepFace.verify(
            image1_path,
            image2_path,
            model_name="Facenet",
            detector_backend="opencv",  
            enforce_detection=False
        )
        return result.get("distance", 1.0)
    except Exception as e:
        log(f"[ERROR] Error al comparar rostros con DeepFace: {e}")
        return 1.0


def get_face_embedding(image_np):
    """
    Devuelve el embedding (vector facial) del primer rostro detectado.
    """
    reps = DeepFace.represent(
        img_path=image_np,            
        model_name="Facenet",
        detector_backend="opencv",     
        enforce_detection=True
    )
    if isinstance(reps, list) and len(reps) > 0 and "embedding" in reps[0]:
        emb = reps[0]["embedding"]
        return [float(x) for x in emb]
    raise ValueError("No se obtuvo embedding de la imagen.")

# ============================================================
# ENDPOINT 1: VALIDAR ROSTRO EN IMAGEN
# ============================================================
@app.route('/validate_face_from_drive', methods=['POST'])
def validate_face_from_drive():
    """
    Verifica si una imagen contiene al menos un rostro válido.
    """
    try:
        log("\n[INFO] Nueva solicitud /validate_face_from_drive")
        if VERBOSE:
            dlog(f"[DATA RAW] {request.data}")

        data = request.get_json(force=True)
        file_id = data.get('file_id')

        if not file_id:
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"valid": False, "message": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        temp_path = f"/tmp/temp_validate_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_path, image)

        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend="opencv", 
            enforce_detection=True
        )

        try:
            os.remove(temp_path)
        except Exception:
            pass

        if len(faces) > 0:
            log("[RESULTADO] Rostro detectado ✅")
            result = {"valid": True, "message": "La imagen contiene al menos un rostro válido."}
        else:
            log("[RESULTADO] Sin rostros detectados ❌")
            result = {"valid": False, "message": "No se detectaron rostros en la imagen."}

        gc.collect()
        return Response(json.dumps(result), mimetype="application/json", status=200)

    except Exception as e:
        log(f"[ERROR] /validate_face_from_drive -> {e}")
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# ENDPOINT 2: COMPARAR ROSTROS Y ACTUALIZAR BD
# ============================================================
@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros y actualiza el score de similitud en la base de datos.
    Devuelve SOLO {"similarity_score": <float>} para AppSheet.
    """
    try:
        log("\n[INFO] Nueva solicitud /compare_faces_from_drive")
        if VERBOSE:
            dlog(f"[HEADERS] {dict(request.headers)}")
            dlog(f"[DATA RAW] {request.data}")

        data = request.get_json(force=True)
        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')
        clock_id = data.get('clock_id')

        if not file_id1 or not file_id2 or not clock_id:
            return Response(json.dumps({"error": "Faltan parámetros"}), mimetype="application/json", status=400)

        image1 = download_image_from_drive(file_id1)
        image2 = download_image_from_drive(file_id2)
        if image1 is None or image2 is None:
            return Response(json.dumps({"error": "No se pudieron descargar las imágenes"}), mimetype="application/json", status=400)

        req_id = str(uuid.uuid4())
        temp1 = f"/tmp/temp_image1_{req_id}.jpg"
        temp2 = f"/tmp/temp_image2_{req_id}.jpg"
        cv2.imwrite(temp1, image1)
        cv2.imwrite(temp2, image2)

        distance = compare_faces(temp1, temp2)
        similarity_score = round(1 - distance, 5)

        if similarity_score >= 0.5:
            log(f"[RESULTADO] Rostros similares ✅ | Score: {similarity_score}")
        else:
            log(f"[RESULTADO] Rostros diferentes ❌ | Score: {similarity_score}")

        # Actualización en BD
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                update_query = """
                    UPDATE rhClockV
                    SET ckBiometrics = %s
                    WHERE ClockID = %s;
                """
                dlog(f"[DB] UPDATE rhClockV SET ckBiometrics={similarity_score} WHERE ClockID={clock_id}")
                cursor.execute(update_query, (similarity_score, clock_id))
                conn.commit()
                cursor.close()
                conn.close()
                log(f"[DB] ✅ ckBiometrics actualizado correctamente en ClockID={clock_id}")
            except Exception as dbe:
                log(f"[DB][ERROR] Falló el UPDATE: {dbe}")
        else:
            log("[WARN] No se pudo conectar a la base de datos")

        try:
            os.remove(temp1); os.remove(temp2)
        except Exception as e:
            dlog(f"[WARN] No se pudo eliminar temporal: {e}")

        gc.collect()
        return Response(json.dumps({"similarity_score": similarity_score}), mimetype="application/json", status=200)

    except Exception as e:
        log(f"[ERROR] /compare_faces_from_drive -> {e}")
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# ENDPOINT 3: OBTENER EMBEDDING FACIAL
# ============================================================
@app.route('/face_embedding', methods=['POST'])
def face_embedding():
    """
    Obtiene el embedding facial usando DeepFace (Facenet).
    - JSON: {"file_id": "...", "staff_id": "..." (opcional)}
    - Si trae staff_id, lo guarda en rhStaff.FaceEmbedding (JSON).
    """
    try:
        log("\n[INFO] Solicitud /face_embedding")
        data = request.get_json(force=True)

        file_id = data.get("file_id")
        staff_id = data.get("staff_id")

        if not file_id:
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"error": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        embedding = get_face_embedding(image)
        dlog(f"[OK] Embedding generado. Dim={len(embedding)}")

        if staff_id:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    emb_json = json.dumps(embedding, ensure_ascii=False)
                    sql = "UPDATE rhStaff SET FaceEmbedding = %s WHERE StaffID = %s"
                    dlog(f"[DB] Guardando embedding para StaffID={staff_id}")
                    cursor.execute(sql, (emb_json, staff_id))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    log(f"[DB] ✅ FaceEmbedding actualizado")
                except Exception as dbe:
                    log(f"[DB][ERROR] No se pudo guardar FaceEmbedding: {dbe}")

        gc.collect()
        return Response(json.dumps({"embedding": embedding}), mimetype="application/json", status=200)

    except Exception as e:
        log(f"[ERROR] /face_embedding -> {e}")
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# EJECUCIÓN DEL SERVIDOR
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    log(f"[INFO] Iniciando servidor en puerto {port}")
    app.run(threaded=True, debug=True, host='0.0.0.0', port=port)
