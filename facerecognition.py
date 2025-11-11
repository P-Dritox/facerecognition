import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
from mysql.connector import pooling
import gc
import sys
from threading import Lock, Semaphore
import random

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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

_MODEL_CACHE = {"facenet": None}
_MODEL_LOCK = Lock()
_INFER_SEM = Semaphore(int(os.getenv("MAX_INFER_CONCURRENCY", "2")))

def get_model():
    with _MODEL_LOCK:
        if _MODEL_CACHE["facenet"] is None:
            _MODEL_CACHE["facenet"] = DeepFace.build_model("Facenet")
    return _MODEL_CACHE["facenet"]

_DBPOOL = None

def _init_dbpool():
    global _DBPOOL
    if _DBPOOL is not None:
        return
    ca_path = "/etc/secrets/server-ca.pem"
    if not os.path.exists(ca_path):
        log_e(f"[DB] Certificado no encontrado en {ca_path}")
        return
    _DBPOOL = pooling.MySQLConnectionPool(
        pool_name="kaizen_pool",
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
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

def get_db_connection(retries: int = 3, delay: float = 0.4):
    _init_dbpool()
    if _DBPOOL is None:
        return None
    last_err = None
    for attempt in range(retries + 1):
        try:
            return _DBPOOL.get_connection()
        except Exception as err:
            last_err = err
            sleep_s = delay * (2 ** attempt) + random.uniform(0, 0.2)
            log_w(f"[DB] intento {attempt+1}/{retries+1} fallo: {err} (sleep {sleep_s:.2f}s)")
            time.sleep(sleep_s)
    log_e(f"[DB] No se pudo conectar (ultimo error): {last_err}")
    return None

def persist_score(clock_id: str, score: float) -> bool:
    try:
        conn = get_db_connection()
        if not conn:
            log.warning("[DB] sin conexión para persistir ckBiometrics=%s (ClockID=%s)", score, clock_id)
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE rhClockV SET ckBiometrics = %s WHERE ClockID = %s;",
                (score, clock_id)
            )
            conn.commit()
            ok = cur.rowcount >= 1
            return ok
        finally:
            try: cur.close()
            except Exception: pass
            try: conn.close()
            except Exception: pass
    except Exception as e:
        log.warning("[DB] fallo persist_score(%s): %s", clock_id, e)
        return False

_RS = requests.Session()
_RS.headers.update({"User-Agent": "Kaizen-FaceRec/1.0"})
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_RS.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.2, status_forcelist=(429, 500, 502, 503, 504))))

def download_image_from_drive(file_id: str):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = _RS.get(url, stream=True, timeout=(5, 25))
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            return None
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def resize_max_dim(img: np.ndarray, max_dim: int = 720) -> np.ndarray:
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

def get_face_embedding(image_np: np.ndarray):
    get_model()
    with _INFER_SEM:
        reps = DeepFace.represent(
            img_path=image_np,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )
    if isinstance(reps, list) and reps and "embedding" in reps[0]:
        emb = reps[0]["embedding"]
        return [float(x) for x in emb]
    raise ValueError("No se obtuvo embedding.")

def _embedding_from_json(val):
    try:
        if isinstance(val, list):
            return [float(x) for x in val]
        if isinstance(val, str):
            arr = json.loads(val)
            return [float(x) for x in arr] if isinstance(arr, list) else None
        return None
    except Exception:
        return None

_STAFF_EMB_CACHE = {}
_STAFF_EMB_TTL = int(os.getenv("STAFF_EMB_TTL_SEC", "3600"))

def _staff_cache_get(staff_id: str):
    if not staff_id:
        return None
    item = _STAFF_EMB_CACHE.get(staff_id)
    if not item:
        return None
    emb, ts = item
    if time.time() - ts > _STAFF_EMB_TTL:
        _STAFF_EMB_CACHE.pop(staff_id, None)
        return None
    return emb

def _staff_cache_put(staff_id: str, emb):
    if staff_id and isinstance(emb, list) and emb:
        _STAFF_EMB_CACHE[staff_id] = (emb, time.time())

def cosine_similarity(a, b):
    try:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        sim = float(np.dot(va, vb) / (na * nb))
        if sim < 0.0: sim = 0.0
        if sim > 1.0: sim = 1.0
        return sim
    except Exception:
        return 0.0

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route('/validate_face_from_drive', methods=['POST'])
def validate_face_from_drive():
    try:
        data = request.get_json(force=True)
        file_id = (data or {}).get('file_id')
        if not file_id:
            return Response(json.dumps({"valid": False, "reason": "missing_file_id"}), mimetype="application/json", status=200)
        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"valid": False, "reason": "download_failed"}), mimetype="application/json", status=200)
        image = resize_max_dim(image, 720)
        faces = DeepFace.extract_faces(img_path=image, detector_backend="opencv", enforce_detection=False)
        ok = len(faces) > 0
        gc.collect()
        return Response(json.dumps({"valid": ok}), mimetype="application/json", status=200)
    except Exception:
        gc.collect()
        return Response(json.dumps({"valid": False, "reason": "server_exception"}), mimetype="application/json", status=200)

@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    req_id = uuid.uuid4().hex[:8]
    try:
        data = request.get_json(force=True) or {}
        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')
        clock_id = data.get('clock_id')
        if not file_id1 or not file_id2 or not clock_id:
            saved = persist_score(clock_id, 0.0) if clock_id else False
            return Response(json.dumps({"similarity_score": 0.0, "db_saved": bool(saved), "reason": "missing_params"}), mimetype="application/json", status=200)
        img1 = download_image_from_drive(file_id1)
        img2 = download_image_from_drive(file_id2)
        if img1 is None or img2 is None:
            saved = persist_score(clock_id, 0.0)
            return Response(json.dumps({"similarity_score": 0.0, "db_saved": bool(saved), "reason": "download_failed"}), mimetype="application/json", status=200)
        img1 = resize_max_dim(img1, 720)
        img2 = resize_max_dim(img2, 720)
        tmp1 = f"/tmp/df_{req_id}_1.jpg"
        tmp2 = f"/tmp/df_{req_id}_2.jpg"
        cv2.imwrite(tmp1, img1)
        cv2.imwrite(tmp2, img2)
        try:
            get_model()
            with _INFER_SEM:
                result = DeepFace.verify(
                    img1_path=tmp1,
                    img2_path=tmp2,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )
            distance = float(result.get("distance", 1.0))
            similarity = 1.0 - distance
            if similarity < 0.0: similarity = 0.0
            if similarity > 1.0: similarity = 1.0
            saved = persist_score(clock_id, similarity)
            return Response(json.dumps({"similarity_score": round(similarity, 5), "db_saved": bool(saved)}), mimetype="application/json", status=200)
        except Exception:
            saved = persist_score(clock_id, 0.0)
            return Response(json.dumps({"similarity_score": 0.0, "db_saved": bool(saved), "reason": "df_error"}), mimetype="application/json", status=200)
        finally:
            for p in (tmp1, tmp2):
                try: os.remove(p)
                except Exception: pass
    except Exception:
        try:
            data = request.get_json(silent=True) or {}
            clock_id = data.get('clock_id')
            if clock_id:
                persist_score(clock_id, 0.0)
        except Exception:
            pass
        return Response(json.dumps({"similarity_score": 0.0, "db_saved": False, "reason": "server_exception"}), mimetype="application/json", status=200)

@app.route('/face_embedding', methods=['POST'])
def face_embedding():
    try:
        data = request.get_json(force=True) or {}
        file_id = data.get("file_id")
        staff_id = data.get("staff_id")
        if not file_id:
            return Response(json.dumps({"embedding": None, "db_saved": False, "reason": "missing_file_id"}), mimetype="application/json", status=200)
        img = download_image_from_drive(file_id)
        if img is None:
            return Response(json.dumps({"embedding": None, "db_saved": False, "reason": "download_failed"}), mimetype="application/json", status=200)
        img = resize_max_dim(img, 720)
        try:
            emb = get_face_embedding(img)
        except Exception:
            return Response(json.dumps({"embedding": None, "db_saved": False, "reason": "no_face_or_df_error"}), mimetype="application/json", status=200)
        db_saved = False
        if staff_id:
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    emb_json = json.dumps(emb, ensure_ascii=False)
                    cur.execute("UPDATE rhStaff SET FaceEmbedding = %s WHERE StaffID = %s", (emb_json, staff_id))
                    conn.commit()
                    db_saved = (cur.rowcount >= 1)
                    _staff_cache_put(staff_id, emb)
                except Exception:
                    pass
                finally:
                    try: cur.close()
                    except: pass
                    try: conn.close()
                    except: pass
        gc.collect()
        return Response(json.dumps({"embedding": emb, "db_saved": bool(db_saved)}), mimetype="application/json", status=200)
    except Exception:
        gc.collect()
        return Response(json.dumps({"embedding": None, "db_saved": False, "reason": "server_exception"}), mimetype="application/json", status=200)

@app.route('/face_validation', methods=['POST'])
def face_validation():
    try:
        data = request.get_json(force=True) or {}
        staff_id = data.get("staff_id")
        clock_id = data.get("clock_id")
        staff_embedding_body = data.get("staff_embedding")
        st_image = data.get("stImage")
        ck_image = data.get("ckImage")

        if not staff_id or not clock_id:
            return Response(json.dumps({
                "similarity_score": 0.0, "db_saved": False,
                "updated_staff_embedding": False, "reason": "missing_ids"
            }), mimetype="application/json", status=200)

        staff_emb = None
        updated_staff = False

        # 1) Staff: usar embedding si viene; si no, construirlo desde stImage y guardarlo
        if staff_embedding_body:
            staff_emb = _embedding_from_json(staff_embedding_body)
        else:
            if not st_image:
                saved = persist_score(clock_id, 0.0)
                return Response(json.dumps({
                    "similarity_score": 0.0, "db_saved": bool(saved),
                    "updated_staff_embedding": False, "reason": "missing_stImage"
                }), mimetype="application/json", status=200)

            img_staff = download_image_from_drive(st_image)
            if img_staff is None:
                saved = persist_score(clock_id, 0.0)
                return Response(json.dumps({
                    "similarity_score": 0.0, "db_saved": bool(saved),
                    "updated_staff_embedding": False, "reason": "download_stImage_failed"
                }), mimetype="application/json", status=200)

            img_staff = resize_max_dim(img_staff, 720)
            try:
                staff_emb = get_face_embedding(img_staff)
            except Exception:
                saved = persist_score(clock_id, 0.0)
                return Response(json.dumps({
                    "similarity_score": 0.0, "db_saved": bool(saved),
                    "updated_staff_embedding": False, "reason": "stImage_no_face"
                }), mimetype="application/json", status=200)

            # Guardar FaceEmbedding recién creado
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE rhStaff SET FaceEmbedding = %s WHERE StaffID = %s",
                        (json.dumps(staff_emb, ensure_ascii=False), staff_id)
                    )
                    conn.commit()
                    updated_staff = (cur.rowcount >= 1)
                finally:
                    try: cur.close()
                    except: pass
                    try: conn.close()
                    except: pass

        # 2) Marca: siempre desde ckImage (si falta, no podemos validar)
        if not ck_image:
            saved = persist_score(clock_id, 0.0)
            return Response(json.dumps({
                "similarity_score": 0.0, "db_saved": bool(saved),
                "updated_staff_embedding": bool(updated_staff), "reason": "missing_ckImage"
            }), mimetype="application/json", status=200)

        img_mark = download_image_from_drive(ck_image)
        if img_mark is None:
            saved = persist_score(clock_id, 0.0)
            return Response(json.dumps({
                "similarity_score": 0.0, "db_saved": bool(saved),
                "updated_staff_embedding": bool(updated_staff), "reason": "download_ckImage_failed"
            }), mimetype="application/json", status=200)

        img_mark = resize_max_dim(img_mark, 720)
        try:
            mark_emb = get_face_embedding(img_mark)
        except Exception:
            saved = persist_score(clock_id, 0.0)
            return Response(json.dumps({
                "similarity_score": 0.0, "db_saved": bool(saved),
                "updated_staff_embedding": bool(updated_staff), "reason": "ckImage_no_face"
            }), mimetype="application/json", status=200)

        # 3) Similaridad y persistencia
        sim = cosine_similarity(staff_emb, mark_emb)
        sim = max(0.0, min(1.0, float(sim)))

        saved = persist_score(clock_id, sim)
        return Response(json.dumps({
            "similarity_score": round(sim, 5),
            "db_saved": bool(saved),
            "updated_staff_embedding": bool(updated_staff)
        }), mimetype="application/json", status=200)

    except Exception:
        try:
            data = request.get_json(silent=True) or {}
            clock_id = data.get('clock_id')
            if clock_id:
                persist_score(clock_id, 0.0)
        except Exception:
            pass
        return Response(json.dumps({
            "similarity_score": 0.0, "db_saved": False,
            "updated_staff_embedding": False, "reason": "server_exception"
        }), mimetype="application/json", status=200)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    log_i(f"[INFO] Iniciando servidor en puerto {port}")
    app.run(threaded=True, debug=False, host='0.0.0.0', port=port)
