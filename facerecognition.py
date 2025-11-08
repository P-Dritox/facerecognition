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
import base64  # <--- añadido

# ============================================================
# CONFIGURACIÓN BASE
# ============================================================
app = Flask(__name__)
CORS(app)

# TensorFlow: Forzar CPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# CONEXIÓN A BASE DE DATOS
# ============================================================
def get_db_connection():
    """
    Conexión segura a MySQL en Google Cloud SQL.
    Usa solo el certificado server-ca.pem y evita dependencias del sistema.
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
            ssl_verify_cert=True
        )
        print("[DB] Conexión SSL establecida correctamente ✅", flush=True)
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
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        return result.get("distance", 1.0)
    except Exception as e:
        print(f"[ERROR] Error al comparar rostros con DeepFace: {e}", flush=True)
        return 1.0


def get_face_embedding(image_np):
    """
    Devuelve el embedding (lista de floats) del primer rostro detectado en image_np.
    Lanza ValueError si no se detecta rostro.
    """
    reps = DeepFace.represent(
        img_path=image_np,         # admite NumPy directamente
        model_name="Facenet",      # vector de 128 dims
        detector_backend="mtcnn",
        enforce_detection=True,
        align=True
    )
    # represent() retorna lista de dicts (uno por rostro detectado)
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
        print("\n[INFO] Nueva solicitud recibida para validar rostro", flush=True)
        print(f"[DATA RAW] {request.data}", flush=True)

        data = request.get_json(force=True)
        file_id = data.get('file_id')

        if not file_id:
            print("[ERROR] Falta file_id en la solicitud", flush=True)
            return Response(json.dumps({"error": "Debe incluir file_id"}), mimetype="application/json", status=400)

        image = download_image_from_drive(file_id)
        if image is None:
            return Response(json.dumps({"valid": False, "message": "No se pudo descargar la imagen"}), mimetype="application/json", status=400)

        temp_path = f"/tmp/temp_validate_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_path, image)

        try:
            faces = DeepFace.extract_faces(img_path=temp_path, detector_backend="mtcnn", enforce_detection=True)
            if len(faces) > 0:
                print("[RESULTADO] Rostro detectado ✅", flush=True)
                result = {"valid": True, "message": "La imagen contiene al menos un rostro válido."}
            else:
                print("[RESULTADO] Sin rostros detectados ❌", flush=True)
                result = {"valid": False, "message": "No se detectaron rostros en la imagen."}
        except Exception as e:
            print(f"[WARN] No se detectó un rostro: {e}", flush=True)
            result = {"valid": False, "message": "No se detectó un rostro en la imagen."}

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return Response(json.dumps(result), mimetype="application/json", status=200)

    except Exception as e:
        print(f"[ERROR] Error general en validate_face_from_drive: {e}", flush=True)
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


# ============================================================
# ENDPOINT 2: COMPARAR ROSTROS Y ACTUALIZAR BD
# ============================================================
@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros (file_id1, file_id2) y actualiza ckBiometrics en la BD.
    """
    try:
        print("\n[INFO] Nueva solicitud recibida desde AppSheet", flush=True)
        print(f"[HEADERS] {dict(request.headers)}", flush=True)
        print(f"[DATA RAW] {request.data}", flush=True)

        data = request.get_json(force=True)
        print(f"[JSON RECIBIDO] {data}", flush=True)

        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')
        clock_id = data.get('clock_id')
        staff_id = data.get('staff_id')

        if not file_id1 or not file_id2 or not clock_id:
            print("[ERROR] Faltan parámetros obligatorios (file_id1, file_id2, clock_id)", flush=True)
            return Response(json.dumps({"error": "Faltan parámetros"}), mimetype="application/json", status=400)

        image1 = download_image_from_drive(file_id1)
        image2 = download_image_from_drive(file_id2)
        if image1 is None or image2 is None:
            print("[ERROR] No se pudieron descargar las imágenes", flush=True)
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
            print("[DB] Conexión establecida ✅", flush=True)
            cursor = conn.cursor()
            update_query = """
                UPDATE rhClockV
                SET ckBiometrics = %s
                WHERE ClockID = %s;
            """
            print(f"[DB] Ejecutando: UPDATE rhClockV SET ckBiometrics={similarity_score} WHERE ClockID={clock_id}", flush=True)
            cursor.execute(update_query, (similarity_score, clock_id))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"[DB] ✅ ckBiometrics actualizado correctamente en ClockID={clock_id}", flush=True)
        else:
            print("[WARN] No se pudo conectar a la base de datos", flush=True)

        for temp in [temp1, temp2]:
            try:
                os.remove(temp)
            except Exception as e:
                print(f"[WARN] No se pudo eliminar {temp}: {e}", flush=True)

        return Response(json.dumps({"similarity_score": similarity_score}), mimetype="application/json", status=200)

    except Exception as e:
        print(f"[ERROR] Error general en compare_faces_from_drive: {e}", flush=True)
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


# ============================================================
# ENDPOINT 3: EMBEDDING FACIAL (OPCIONALMENTE PERSISTE EN BD)
# ============================================================
@app.route('/face_embedding', methods=['POST'])
def face_embedding():
    """
    Obtiene el embedding facial usando DeepFace (Facenet).

    Formas de uso:
    - multipart/form-data con un archivo en el campo 'image'
    - JSON: {"file_id": "<id de Google Drive>"}  (o "image_base64": "data:image/..;base64,....")

    Comportamiento:
    - Si SOLO envías imagen (sin staff_id), devuelve el embedding como STRING (valores separados por coma).
    - Si envías staff_id, guarda el embedding en bdKaizen.rhStaff.FaceEmbedding (JSON) y responde en JSON.
    """
    try:
        print("\n[INFO] Solicitud /face_embedding", flush=True)

        image_np = None
        staff_id = None

        # 1) multipart/form-data
        if 'image' in request.files:
            file = request.files['image']
            raw = np.frombuffer(file.read(), dtype=np.uint8)
            image_np = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            staff_id = request.form.get('staff_id')

        else:
            # 2) JSON
            data = request.get_json(silent=True) or {}
            staff_id = data.get('staff_id')

            # file_id de Drive
            file_id = data.get('file_id')
            if file_id and image_np is None:
                image_np = download_image_from_drive(file_id)

            # image_base64 (opcional)
            if image_np is None and data.get('image_base64'):
                b64 = data['image_base64']
                if ',' in b64:  # soporta data URL
                    b64 = b64.split(',', 1)[1]
                raw = base64.b64decode(b64)
                image_np = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

        if image_np is None:
            return Response(
                json.dumps({"error": "Debes enviar 'image' (multipart), 'file_id' (Drive) o 'image_base64' (JSON)"}),
                mimetype="application/json", status=400
            )

        # 1) Obtener embedding
        embedding = get_face_embedding(image_np)  # lista de floats
        print(f"[OK] Embedding generado. Dim={len(embedding)}", flush=True)

        # 2) ¿Guardar en BD?
        if staff_id:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    emb_json = json.dumps(embedding, ensure_ascii=False)
                    sql = """
                        UPDATE rhStaff
                           SET FaceEmbedding = %s
                         WHERE StaffID = %s
                    """
                    params = (emb_json, staff_id)
                    print(f"[DB] Guardando embedding en rhStaff (StaffID={staff_id})", flush=True)
                    cursor.execute(sql, params)
                    conn.commit()
                    print("[DB] ✅ FaceEmbedding actualizado", flush=True)
                except Exception as dbe:
                    print(f"[DB][ERROR] No se pudo guardar FaceEmbedding: {dbe}", flush=True)
                finally:
                    try:
                        cursor.close()
                        conn.close()
                    except Exception:
                        pass

            # Respuesta JSON cuando se guarda
            result = {
                "model": "Facenet",
                "dimension": len(embedding),
                "embedding": embedding,
                "saved_for_staff_id": staff_id
            }
            return Response(json.dumps(result), mimetype="application/json", status=200)

        # 3) Si NO hay staff_id: devolver como STRING (coma-separado)
        emb_str = ",".join(str(x) for x in embedding)
        return Response(emb_str, mimetype="text/plain", status=200)

    except Exception as e:
        msg = str(e)
        print(f"[ERROR] /face_embedding -> {msg}", flush=True)
        status = 422 if "face" in msg.lower() or "rostro" in msg.lower() else 500
        return Response(json.dumps({"error": msg}), mimetype="application/json", status=status)



# ============================================================
# EJECUCIÓN DEL SERVIDOR
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"[INFO] Iniciando servidor en puerto {port}", flush=True)
    app.run(threaded=True, debug=True, host='0.0.0.0', port=port)
