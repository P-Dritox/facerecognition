from flask import Flask, request, Response
import requests
import os
import cv2
import numpy as np
from deepface import DeepFace
from flask_cors import CORS
import uuid
import json
import mysql.connector
import base64
import gc

# ============================================================
# CONFIGURACIÓN BASE
# ============================================================
app = Flask(__name__)
CORS(app)

# TensorFlow: Forzar uso de CPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# PRECARGA DE MODELO (vía API pública de DeepFace)
# ============================================================
print("[INIT] Precargando modelo Facenet...", flush=True)
facenet_model = DeepFace.build_model("Facenet")  # ✅ reemplaza import interno
print("[INIT] Modelo cargado correctamente ✅", flush=True)

# ============================================================
# CONEXIÓN A BASE DE DATOS
# ============================================================
def get_db_connection():
    """
    Conexión segura a MySQL (GCP) usando certificado SSL.
    """
    try:
        ca_path = "/etc/secrets/server-ca.pem"
        if not os.path.exists(ca_path):
            print(f"[ERROR] Certificado no encontrado en {ca_path}", flush=True)
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
        print(f"[ERROR] Error MySQL: {err}", flush=True)
        return None
    except Exception as e:
        print(f"[ERROR] Error general de conexión: {e}", flush=True)
        return None

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def download_image_from_drive(file_id):
    """
    Descarga una imagen desde Google Drive y la devuelve como matriz.
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[INFO] Descargando imagen desde: {url}", flush=True)
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"[ERROR] No se pudo descargar la imagen (HTTP {response.status_code}) con file_id: {file_id}", flush=True)
            return None
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Error al descargar la imagen: {e}", flush=True)
        return None

def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes de rostros y devuelve la distancia.
    """
    try:
        result = DeepFace.verify(
            image1_path,
            image2_path,
            model_name="Facenet",
            model=facenet_model,        # ✅ reutiliza el modelo precargado
            detector_backend="opencv",  # liviano
            enforce_detection=False
        )
        return result.get("distance", 1.0)
    except Exception as e:
        print(f"[ERROR] Error al comparar rostros con DeepFace: {e}", flush=True)
        return 1.0

def get_face_embedding(image_np):
    """
    Devuelve el embedding (vector facial) del primer rostro detectado.
    """
    reps = DeepFace.represent(
        img_path=image_np,
        model_name="Facenet",
        model=facenet_model,        # ✅ reutiliza el modelo
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
        print("\n[INFO] Nueva solicitud /validate_face_from_drive", flush=True)
        print(f"[DATA RAW] {request.data}", flush=True)

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
        os.remove(temp_path)

        if len(faces) > 0:
            print("[RESULTADO] Rostro detectado ✅", flush=True)
            result = {"valid": True, "message": "La imagen contiene al menos un rostro válido."}
        else:
            print("[RESULTADO] Sin rostros detectados ❌", flush=True)
            result = {"valid": False, "message": "No se detectaron rostros en la imagen."}

        gc.collect()
        return Response(json.dumps(result), mimetype="application/json", status=200)

    except Exception as e:
        print(f"[ERROR] /validate_face_from_drive -> {e}", flush=True)
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# ENDPOINT 2: COMPARAR ROSTROS Y ACTUALIZAR BD
# ============================================================
@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros y actualiza el score de similitud en la base de datos.
    """
    try:
        print("\n[INFO] Nueva solicitud /compare_faces_from_drive", flush=True)
        print(f"[HEADERS] {dict(request.headers)}", flush=True)
        print(f"[DATA RAW] {request.data}", flush=True)

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
            print(f"[RESULTADO] Rostros similares ✅ | Score: {similarity_score}", flush=True)
        else:
            print(f"[RESULTADO] Rostros diferentes ❌ | Score: {similarity_score}", flush=True)

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            update_query = """
                UPDATE rhClockV
                   SET ckBiometrics = %s
                 WHERE ClockID = %s;
            """
            cursor.execute(update_query, (similarity_score, clock_id))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"[DB] ✅ ckBiometrics actualizado correctamente en ClockID={clock_id}", flush=True)
        else:
            print("[WARN] No se pudo conectar a la base de datos", flush=True)

        os.remove(temp1)
        os.remove(temp2)
        gc.collect()

        return Response(json.dumps({"similarity_score": similarity_score}), mimetype="application/json", status=200)

    except Exception as e:
        print(f"[ERROR] /compare_faces_from_drive -> {e}", flush=True)
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# ENDPOINT 3: OBTENER EMBEDDING FACIAL
# ============================================================
@app.route('/face_embedding', methods=['POST'])
def face_embedding():
    """
    Obtiene el embedding facial usando DeepFace (Facenet).
    """
    try:
        print("\n[INFO] Solicitud /face_embedding", flush=True)
        data = request.get_json(force=True)

        file_id = data.get("file_id")
        staff_id = data.get("staff_id")

        if not file_id:
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"error": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        embedding = get_face_embedding(image)
        print(f"[OK] Embedding generado. Dim={len(embedding)}", flush=True)

        if staff_id:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    emb_json = json.dumps(embedding, ensure_ascii=False)
                    sql = "UPDATE rhStaff SET FaceEmbedding = %s WHERE StaffID = %s"
                    cursor.execute(sql, (emb_json, staff_id))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    print(f"[DB] ✅ FaceEmbedding actualizado para StaffID={staff_id}", flush=True)
                except Exception as dbe:
                    print(f"[DB][ERROR] No se pudo guardar FaceEmbedding: {dbe}", flush=True)

        gc.collect()
        return Response(json.dumps({"embedding": embedding}), mimetype="application/json", status=200)

    except Exception as e:
        print(f"[ERROR] /face_embedding -> {e}", flush=True)
        gc.collect()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# ============================================================
# EJECUCIÓN DEL SERVIDOR
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"[INFO] Iniciando servidor en puerto {port}", flush=True)
    app.run(threaded=True, debug=True, host='0.0.0.0', port=port)
