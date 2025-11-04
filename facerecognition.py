from flask import Flask, request, jsonify
import requests
import os
import cv2
import numpy as np
import uuid
from deepface import DeepFace
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def download_image_from_drive(file_id):
    """
    Descarga una imagen desde Google Drive y la devuelve como una matriz en memoria.
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Descargando imagen desde: {url}")
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"Error al descargar la imagen con file_id: {file_id}")
            return None
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al descargar la imagen: {e}")
        return None

def is_valid_image_path(image_path):
    """
    Verifica si la ruta del archivo de imagen es válida.
    """
    return isinstance(image_path, str) and os.path.exists(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png'))

def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes usando DeepFace y retorna una medida de similitud.
    """
    try:
        if not is_valid_image_path(image1_path) or not is_valid_image_path(image2_path):
            print("Error: Una o ambas rutas de imágenes no son válidas.")
            return False, 1

        result = DeepFace.verify(image1_path, image2_path, model_name="Facenet", enforce_detection=False, detector_backend="mtcnn")
        return result.get("verified", False), result.get("distance", 1)
    except Exception as e:
        print(f"Error al comparar rostros con DeepFace: {e}")
        return False, 1

@app.route('/validate_face_from_drive', methods=['POST'])
def validate_face_from_drive():
    """
    Verifica si una imagen contiene un rostro válido utilizando su file_id desde Google Drive.
    """
    try:
        print("Solicitud recibida para validar rostro")
        print(f"Headers: {request.headers}")
        print(f"Data: {request.data}")

        data = request.get_json()
        print(f"JSON recibido: {data}")

        file_id = data.get('file_id')

        if not file_id:
            print("Falta file_id en la solicitud")
            return jsonify({"error": "Por favor, proporciona el file_id."}), 400

        image = download_image_from_drive(file_id)

        if image is None:
            return jsonify({"error": "No se pudo descargar la imagen."}), 400

        # Crear un nombre único temporal
        request_id = str(uuid.uuid4())
        temp_path = f"/tmp/temp_validate_{request_id}.jpg"
        cv2.imwrite(temp_path, image)

        try:
            faces = DeepFace.extract_faces(img_path=temp_path, detector_backend="mtcnn", enforce_detection=True)
            os.remove(temp_path)
            if len(faces) > 0:
                return jsonify({
                    "valid": True,
                    "message": "La imagen contiene al menos un rostro válido."
                })
            else:
                return jsonify({
                    "valid": False,
                    "message": "No se detectaron rostros en la imagen."
                })
        except Exception as e:
            print(f"No se detectó un rostro: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                "valid": False,
                "message": "No se detectó un rostro en la imagen.",
                "error": str(e)
            })

    except Exception as e:
        print(f"Error en la validación de rostro: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare_faces_from_drive', methods=['POST'])
def compare_faces_from_drive():
    """
    Compara dos rostros desde Google Drive utilizando sus file_ids.
    """
    try:
        print("Solicitud recibida desde AppSheet")
        print(f"Headers: {request.headers}")
        print(f"Data: {request.data}")
        
        data = request.get_json()
        print(f"JSON recibido: {data}")

        file_id1 = data.get('file_id1')
        file_id2 = data.get('file_id2')

        if not file_id1 or not file_id2:
            print("Faltan file_id1 o file_id2 en la solicitud")
            return jsonify({"error": "Por favor, proporciona ambos file_ids."}), 400

        image1 = download_image_from_drive(file_id1)
        image2 = download_image_from_drive(file_id2)

        if image1 is None or image2 is None:
            return jsonify({"error": "No se pudo descargar una o ambas imágenes."}), 400

        # Crear nombres únicos por solicitud
        request_id = str(uuid.uuid4())
        temp1 = f"/tmp/temp_image1_{request_id}.jpg"
        temp2 = f"/tmp/temp_image2_{request_id}.jpg"

        cv2.imwrite(temp1, image1)
        cv2.imwrite(temp2, image2)

        attempts = 0
        match, distance = None, None
        while attempts < 2:
            match, distance = compare_faces(temp1, temp2)
            if match:
                break
            attempts += 1

        if os.path.exists(temp1):
            os.remove(temp1)
        if os.path.exists(temp2):
            os.remove(temp2)

        if match is None:
            print("No se pudo realizar la comparación de rostros")
            return jsonify({"error": "No se pudo realizar la comparación."}), 500

        similarity_score = 1 - distance
        message = "Los rostros son similares." if similarity_score >= 0.5 else "Los rostros no son similares."

        return jsonify({
            "match": match,
            "similarity_score": similarity_score,
            "message": message
        })

    except Exception as e:
        print(f"Error en la comparación: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '_main_':
    app.run(threaded=True, debug=True, host='0.0.0.0', port=5000)
