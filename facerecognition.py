# facerecognition.py
import os

# --- Reducir ruido y huella de TF / CPU ANTES de importar DeepFace/TensorFlow ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # fuerza CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # 0=ALL, 1=INFO off, 2=WARNING off, 3=ERROR off
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, request, Response
from flask_cors import CORS
import logging
import uuid
import json
import time
import requests
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
import base64
import gc
import logging, sys


# Silenciar aún más logs de TF/absl
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================
# APP / LOGGING
# ============================================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)
log = logging.getLogger(__name__)


def rid():
    return str(uuid.uuid4())[:8]

def log_i(msg): logging.info(msg)
def log_w(msg): logging.warning(msg)
def log_e(msg): logging.error(msg)
    
# ============================================================
# CARGA LAZY DEL MODELO
# ============================================================
_MODEL_CACHE = {"facenet": None}

def get_model():
    """Carga única del modelo Facenet (por proceso)."""
    if _MODEL_CACHE["facenet"] is None:
        log_i("[INIT] Cargando modelo Facenet (lazy)...")
        _MODEL_CACHE["facenet"] = DeepFace.build_model("Facenet")
        log_i("[INIT] Modelo Facenet listo ✅")
    return _MODEL_CACHE["facenet"]

# ============================================================
# DB
# ============================================================
def get_db_connection(retries: int = 1, delay: float = 0.4):
    """
    Conexión a MySQL (GCP) con SSL CA (solo server). Reintenta brevemente.
    No toca tus variables de entorno.
    """
    ca_path = "/etc/secrets/server-ca.pem"
    if not os.path.exists(ca_path):
        log_e(f"[DB] Certificado no encontrado en {ca_path}")
        return None

    last_err = None
    for attempt in range(retries + 1):
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                database=os.getenv("DB_NAME", "bdKaizen"),
                port=int(os.getenv("DB_PORT", 3306)),
                ssl_ca=ca_path,
                ssl_verify_cert=True,
                autocommit=True,
                connection_timeout=8,
            )
            return conn
        except mysql.connector.Error as err:
            last_err = err
            log_w(f"[DB] intento {attempt+1}/{retries+1} fallo: {err}")
            time.sleep(delay)
        except Exception as e:
            last_err = e
            log_w(f"[DB] intento {attempt+1}/{retries+1} fallo general: {e}")
            time.sleep(delay)
    log_e(f"[DB] No se pudo conectar (ultimo error): {last_err}")
    return None

# ============================================================
# UTIL
# ============================================================

def persist_score(clock_id: str, score: float) -> bool:
    """
    Intenta actualizar ckBiometrics con 'score' para el ClockID dado.
    Devuelve True si actualiza; False si no pudo (sin levantar excepción).
    """
    try:
        conn = get_db_connection()
        if not conn:
            log.warning("[DB] no hay conexión para persistir score=%s (ClockID=%s)", score, clock_id)
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE rhClockV SET ckBiometrics = %s WHERE ClockID = %s;",
                (score, clock_id)
            )
            conn.commit()
            log.info("[DB] ckBiometrics=%s actualizado (ClockID=%s)", score, clock_id)
            return True
        finally:
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        log.warning("[DB] fallo persist_score(%s): %s", clock_id, e)
        return False

def download_image_from_drive(file_id: str):
    """
    Descarga imagen desde Drive a matriz OpenCV (BGR).
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        log_i(f"[DL] GET {url}")
        resp = requests.get(url, stream=True, timeout=(5, 25))
        if resp.status_code != 200:
            log_e(f"[DL] HTTP {resp.status_code} al descargar file_id={file_id}")
            return None
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        log_e(f"[DL] Error download file_id={file_id}: {e}")
        return None

def resize_max_dim(img: np.ndarray, max_dim: int = 720) -> np.ndarray:
    """
    Reduce tamaño para bajar RAM/CPU manteniendo relación de aspecto.
    """
    try:
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_dim:
            return img
        scale = max_dim / float(m)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        return img

def compare_faces(image1_path: str, image2_path: str) -> float:
    """
    Retorna distancia DeepFace (float). similarity = 1 - distance.
    Usamos la API pública estable (sin 'model=').
    """
    try:
        get_model()
        result = DeepFace.verify(
            image1_path,
            image2_path,
            model_name="Facenet",
            detector_backend="opencv", 
            enforce_detection=False
        )
        return float(result.get("distance", 1.0))
    except Exception as e:
        log_e(f"[DF] verify error: {e}")
        return 1.0

def get_face_embedding(image_np: np.ndarray):
    """
    Retorna el embedding del primer rostro (lista de floats).
    """
    # Garantiza modelo cargado
    get_model()

    reps = DeepFace.represent(
        img_path=image_np,               # DeepFace acepta ndarray
        model_name="Facenet",
        detector_backend="opencv",       # Cambiar a "mtcnn" si preferís
        enforce_detection=True
    )
    if isinstance(reps, list) and reps and "embedding" in reps[0]:
        emb = reps[0]["embedding"]
        return [float(x) for x in emb]
    raise ValueError("No se obtuvo embedding.")

# ============================================================
# ENDPOINTS
# ============================================================
@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route('/validate_face_from_drive', methods=['POST'])
def validate_face_from_drive():
    req = rid()
    try:
        data = request.get_json(force=True)
        file_id = (data or {}).get('file_id')
        if not file_id:
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        log_i(f"[{req}] /validate_face_from_drive file_id={file_id}")

        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"valid": False, "message": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        image = resize_max_dim(image, 720)

        tmp = f"/tmp/val_{req}.jpg"
        cv2.imwrite(tmp, image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        faces = []
        try:
            faces = DeepFace.extract_faces(img_path=tmp, detector_backend="opencv", enforce_detection=True)
        finally:
            try: os.remove(tmp)
            except: pass

        ok = len(faces) > 0
        log_i(f"[{req}] validate -> {'rostro ✅' if ok else 'sin rostro ❌'}")
        gc.collect()
        return Response(json.dumps({"valid": ok}), mimetype="application/json", status=200)

    except Exception as e:
        log_e(f"[{req}] validate error: {e}")
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros (file_id1, file_id2) y actualiza ckBiometrics en la BD.
    REGLA: si NO se puede procesar, persiste 0.0 (si hay clock_id) y SIEMPRE responde {"similarity_score": 0.0} con 200.
    """
    req_id = uuid.uuid4().hex[:8]
    try:
        log.info("[%s] Nueva solicitud /compare_faces_from_drive", req_id)
        raw = request.data
        log.info("[%s] [DATA RAW] %s", req_id, raw)

        data = request.get_json(force=True) or {}
        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')
        clock_id = data.get('clock_id')

        if not file_id1 or not file_id2 or not clock_id:
            log.error("[%s] faltan parámetros (file_id1, file_id2, clock_id)", req_id)
            if clock_id:
                persist_score(clock_id, 0.0)
            return Response(json.dumps({"similarity_score": 0.0}), mimetype="application/json", status=200)

        log.info("[%s] [DL] GET https://drive.google.com/uc?id=%s", req_id, file_id1)
        img1 = download_image_from_drive(file_id1)
        log.info("[%s] [DL] GET https://drive.google.com/uc?id=%s", req_id, file_id2)
        img2 = download_image_from_drive(file_id2)
        if img1 is None or img2 is None:
            log.error("[%s] no se pudieron descargar una o ambas imágenes", req_id)
            persist_score(clock_id, 0.0)
            return Response(json.dumps({"similarity_score": 0.0}), mimetype="application/json", status=200)

        tmp1 = f"/tmp/df_{req_id}_1.jpg"
        tmp2 = f"/tmp/df_{req_id}_2.jpg"
        cv2.imwrite(tmp1, img1)
        cv2.imwrite(tmp2, img2)

        try:
            result = DeepFace.verify(
                img1_path=tmp1,
                img2_path=tmp2,
                model_name="Facenet",
                detector_backend="mtcnn",
                enforce_detection=False
            )
            distance = float(result.get("distance", 1.0))
            similarity = 1.0 - distance
            similarity = max(0.0, min(1.0, similarity))

            if similarity >= 0.5:
                log.info("[%s] RESULT -> similares ✅ | score=%s", req_id, similarity)
            else:
                log.info("[%s] RESULT -> diferentes ❌ | score=%s", req_id, similarity)

            persist_score(clock_id, similarity)

            return Response(json.dumps({"similarity_score": round(similarity, 5)}), mimetype="application/json", status=200)

        except Exception as e:
            log.error("[%s] error en DeepFace.verify: %s", req_id, e)
            persist_score(clock_id, 0.0)
            return Response(json.dumps({"similarity_score": 0.0}), mimetype="application/json", status=200)

        finally:
            for p in (tmp1, tmp2):
                try:
                    os.remove(p)
                except Exception:
                    pass

    except Exception as e:
        log.error("[%s] error general: %s", req_id, e)
        try:
            data = request.get_json(silent=True) or {}
            clock_id = data.get('clock_id')
            if clock_id:
                persist_score(clock_id, 0.0)
        except Exception:
            pass
        return Response(json.dumps({"similarity_score": 0.0}), mimetype="application/json", status=200)
        
@app.route('/face_embedding', methods=['POST'])
def face_embedding():
    req = rid()
    try:
        data = request.get_json(force=True)
        file_id = (data or {}).get("file_id")
        staff_id = (data or {}).get("staff_id")

        if not file_id:
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        log_i(f"[{req}] /face_embedding file_id={file_id} staff={staff_id or '-'}")

        img = download_image_from_drive(file_id)
        if img is None:
            return Response(json.dumps({"error": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        img = resize_max_dim(img, 720)

        emb = get_face_embedding(img)
        log_i(f"[{req}] embedding OK dim={len(emb)}")

        if staff_id:
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    emb_json = json.dumps(emb, ensure_ascii=False)
                    cur.execute("UPDATE rhStaff SET FaceEmbedding = %s WHERE StaffID = %s", (emb_json, staff_id))
                    conn.commit()
                    log_i(f"[{req}] DB FaceEmbedding OK (StaffID={staff_id})")
                except Exception as dbe:
                    log_e(f"[{req}] DB FaceEmbedding error: {dbe}")
                finally:
                    try: cur.close()
                    except: pass
                    try: conn.close()
                    except: pass

        gc.collect()
        return Response(json.dumps({"embedding": emb}), mimetype="application/json", status=200)

    except Exception as e:
        log_e(f"[{req}] embedding error: {e}")
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    log_i(f"[INFO] Iniciando servidor en puerto {port}")
    app.run(threaded=True, debug=True, host='0.0.0.0', port=port)
