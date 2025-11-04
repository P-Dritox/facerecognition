from flask import Flask, request, jsonify, Response
import requests
import os
import cv2
import numpy as np
from deepface import DeepFace
from flask_cors import CORS
import uuid
import json

app = Flask(__name__)
CORS(app)

# Configuración TensorFlow / CPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def download_image_from_drive(file_id):
    """
    Descarga una imagen desde Google Drive y la devuelve como matriz en memoria.
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[INFO] Descargando imagen desde: {url}")
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"[ERROR] No se pudo descargar la imagen (HTTP {response.status_code}) con file_id: {file_id}")
            return None
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Error al descargar la imagen: {e}")
        return None


def is_valid_image_path(image_path):
    """
    Verifica si la ruta del archivo de imagen es válida.
    """
    return isinstance(image_path, str) and os.path.exists(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png'))


def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes usando DeepFace y retorna distancia.
    """
    try:
        if not is_valid_image_path(image1_path) or not is_valid_image_path(image2_path):
            print("[ERROR] Una o ambas rutas de imágenes no son válidas.")
            return 1.0

        result = DeepFace.verify(
            image1_path,
            image2_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        distance = result.get("distance", 1.0)
        return distance
    except Exception as e:
        print(f"[ERROR] Error al comparar rostros con DeepFace: {e}")
        return 1.0


# ------------------------------
# ENDPOINT PRINCIPAL
# ------------------------------
@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros desde Google Drive y retorna solo el porcentaje de similitud.
    """
    try:
        print("\n[INFO] Nueva solicitud recibida desde AppSheet")
        print(f"[HEADERS] {request.headers}")
        print(f"[DATA RAW] {request.data}")

        data = request.get_json(force=True)
        print(f"[JSON RECIBIDO] {data}")

        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')

        if not file_id1 or not file_id2:
            print("[ERROR] Faltan file_id1 o file_id2 en la solicitud")
            return Response(json.dumps({"error": "Faltan parámetros"}), mimetype="application/json", status=400)

        # Descargar imágenes
        image1 = download_image_from_drive(file_id1)
        image2 = download_image_from_drive(file_id2)

        if image1 is None or image2 is None:
            print("[ERROR] No se pudieron descargar una o ambas imágenes.")
            return Response(json.dumps({"error": "No se pudieron descargar las imágenes"}), mimetype="application/json", status=400)

        # Archivos temporales únicos
        request_id = str(uuid.uuid4())
        temp1 = f"/tmp/temp_image1_{request_id}.jpg"
        temp2 = f"/tmp/temp_image2_{request_id}.jpg"

        cv2.imwrite(temp1, image1)
        cv2.imwrite(temp2, image2)

        # Comparación de rostros
        distance = compare_faces(temp1, temp2)
        similarity_score = round(1 - distance, 5)

        # Log de resultado
        if similarity_score >= 0.5:
            print(f"[RESULTADO] Rostros similares ✅ | Score: {similarity_score}")
        else:
            print(f"[RESULTADO] Rostros diferentes ❌ | Score: {similarity_score}")

        # Limpieza
        try:
            os.remove(temp1)
            os.remove(temp2)
        except Exception as e:
            print(f"[WARN] No se pudieron eliminar archivos temporales: {e}")

        # Solo retorna el porcentaje
        return Response(json.dumps({"similarity_score": similarity_score}), mimetype="application/json")

    except Exception as e:
        print(f"[ERROR] Error general en la comparación: {e}")
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


# ------------------------------
# INICIO DE LA APLICACIÓN
# ------------------------------
if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0', port=5000)
