from flask import Flask, request, jsonify
import requests
import os
import cv2
import numpy as np
from deepface import DeepFace
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para aceptar solicitudes de cualquier origen

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error al descargar la imagen: {e}")
        return None

def compare_faces(image1_path, image2_path):
    """
    Compara dos imágenes usando DeepFace y retorna una medida de similitud.
    """
    try:
        result = DeepFace.verify(image1_path, image2_path, model_name="SFace", detector_backend="opencv")
        return result["verified"], result["distance"]
    except Exception as e:
        print(f"Error al comparar rostros con DeepFace: {e}")
        return None, None

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

        if image1 is None:
            print(f"No se pudo descargar la imagen con file_id: {file_id1}")
            return jsonify({"error": f"No se pudo descargar la imagen con file_id: {file_id1}"}), 400
        if image2 is None:
            print(f"No se pudo descargar la imagen con file_id: {file_id2}")
            return jsonify({"error": f"No se pudo descargar la imagen con file_id: {file_id2}"}), 400

        # Guardar imágenes temporalmente para DeepFace
        cv2.imwrite("temp_image1.jpg", image1)
        cv2.imwrite("temp_image2.jpg", image2)

        # Comparar rostros
        match, distance = compare_faces("temp_image1.jpg", "temp_image2.jpg")

        if match is None:
            print("No se pudo realizar la comparación de rostros")
            return jsonify({"error": "No se pudo realizar la comparación."}), 500

        print(f"Resultado: match={match}, distance={distance}")
        return jsonify({
            "match": match,
            "similarity_score": 1 - distance,  # Convertir distancia a similitud
            "message": "Los rostros son similares." if match else "Los rostros no son similares."
        })

    except Exception as e:
        print(f"Error en la comparación: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
