from flask import Flask, request, jsonify
import requests
import os
import cv2
import numpy as np
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

def image_has_face(image_path):
    """
    Verifica si una imagen contiene al menos un rostro detectable.
    """
    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend="mtcnn", enforce_detection=True)
        return len(faces) > 0
    except Exception as e:
        print(f"No se detectó un rostro en la imagen: {e}")
        return False

def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes usando DeepFace y retorna una medida de similitud.
    """
    try:
        # Validar las rutas de las imágenes
        if not is_valid_image_path(image1_path) or not is_valid_image_path(image2_path):
            print("Error: Una o ambas rutas de imágenes no son válidas.")
            return False, 1

        # Validar que ambas imágenes contengan un rostro
        if not image_has_face(image1_path) or not image_has_face(image2_path):
            print("Error: Una o ambas imágenes no contienen un rostro válido.")
            return False, 1

        result = DeepFace.verify(image1_path, image2_path, model_name="Facenet", enforce_detection=False, detector_backend="mtcnn")
        return result.get("verified", False), result.get("distance", 1)
    except Exception as e:
        print(f"Error al comparar rostros con DeepFace: {e}")
        return False, 1

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

        # Descargar imágenes desde Google Drive
        image1 = download_image_from_drive(file_id1)
        image2 = download_image_from_drive(file_id2)

        if image1 is None or image2 is None:
            return jsonify({"error": "No se pudo descargar una o ambas imágenes."}), 400

        # Guardar imágenes temporalmente para DeepFace
        cv2.imwrite("temp_image1.jpg", image1)
        cv2.imwrite("temp_image2.jpg", image2)

        attempts = 0
        match, distance = None, None
        while attempts < 2:
            match, distance = compare_faces("temp_image1.jpg", "temp_image2.jpg")
            if match:
                break
            attempts += 1

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

if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0', port=5000)
