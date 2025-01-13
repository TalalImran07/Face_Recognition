from flask import Flask, request, jsonify
from db_connect import Database
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
from landmarks import FacialLandmarkDetector
from grpc_server import serve
import logging
import os
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress Flask development server warning from werkzeug
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

app = Flask(__name__)# Initialize Database object with credentials from environment variables


db = Database(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# Initialize FacialLandmarkDetector with predictor path from environment variable
predictor_path = "./shapes_predictor_68_face_landmarks.dat"
landmark_detector = FacialLandmarkDetector(predictor_path=predictor_path)

# Initialize KNeighborsClassifier
knn_model = None  # Will be trained dynamically


def load_and_train_knn():
    """Load all data from the database and train the KNN model."""
    global knn_model
    landmarks = db.fetch_landmarks()
    if landmarks:
        X = np.array([np.ravel(data[1]) for data in landmarks]).reshape(len(landmarks), -1)
        y = np.array([data[0] for data in landmarks])
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X, y)
        logger.info("KNN model trained successfully with all database data.")
    else:
        knn_model = None
        logger.warning("No data available to train the KNN model.")

@app.before_request
def check_api_key():
    """Check API key in request headers."""
    api_key = os.getenv("API_KEY")
    key = request.headers.get("x-api-key")
    if not key or key != api_key:
        return jsonify({"error": "Unauthorized"}), 401


@app.route('/Save_Image', methods=['POST'])
def save_image():
    if 'image' not in request.files or 'userId' not in request.form:
        return jsonify({"success": False, "message": "No image file or userId provided."}), 400

    image = request.files['image']
    user_id = request.form['userId']

    # Check if userId already exists in the database
    if db.check_user_exists(user_id):
        return jsonify({"success": False, "message": f"User ID {user_id} already exists."}), 400

    file_stream = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(file_stream, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "message": "Invalid image file."}), 400

    face_landmarks = landmark_detector.detect_landmarks(img)
    if face_landmarks is None:
        return jsonify({"success": False, "message": "No face detected in the image."}), 400

    flattened_landmarks = [coord for point in face_landmarks for coord in point]
    if db.save_landmarks(user_id, flattened_landmarks):
        load_and_train_knn()
        return jsonify({"success": True, "message": "Landmarks saved successfully.",
                        "data": {'userId': user_id, 'landmarks': flattened_landmarks}}), 200
    else:
        return jsonify({"success": False, "message": "Failed to save landmarks."}), 500


@app.route('/get_userId', methods=['POST'])
def get_userId():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    image_file = request.files['image']
    file_stream = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_stream, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "message": "Invalid image file."}), 400

    face_landmarks = landmark_detector.detect_landmarks(img)
    if face_landmarks is None:
        return jsonify({"success": False, "message": "No face detected in the image."}), 400

    if knn_model is not None:
        flattened_landmarks = [coord for point in face_landmarks for coord in point]
        face_landmarks_np = np.array(flattened_landmarks).reshape(1, -1)
        try:
            user_id = knn_model.predict(face_landmarks_np)
            return jsonify({"success": True, "message": "User ID retrieved.", "data": {'user_id': int(user_id[0])}}), 200
        except Exception as e:
            logger.error(f"KNN prediction error: {str(e)}")
            return jsonify({"success": False, "message": "Prediction error."}), 500
    else:
        return jsonify({"success": False, "message": "KNN model is not trained."}), 400


def start_grpc_server():
    """Start the gRPC server in a separate thread."""
    serve()


if __name__ == '__main__':
    load_and_train_knn()
    threading.Thread(target=start_grpc_server, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
