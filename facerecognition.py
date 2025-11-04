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
    Retorna una conexión a la base de datos bdKaizen.
    Las variables de entorno se configuran en Render.
    """
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),        # kaizen_api
            password=os.getenv("DB_PASS"),
            database="bdKaizen",
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        print(f"[ERROR] No se pudo conectar a la base de datos: {e}", flush=True)
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
# EJECUCIÓN DEL SERVIDOR
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"[INFO] Iniciando servidor en puerto {port}", flush=True)
    app.run(threaded=True, debug=True, host='0.0.0.0', port=port)
