import grpc
from concurrent import futures
import base64
import numpy as np
import cv2
import logging
import os
from db_connect import Database
from landmarks import FacialLandmarkDetector
from sklearn.neighbors import KNeighborsClassifier
import facial_recognition_pb2
import facial_recognition_pb2_grpc

def check_api_key(context):
    """Check if the API key in metadata matches the environment variable."""
    api_key = os.getenv("API_KEY")  # Fetch API key from environment variable
    metadata = dict(context.invocation_metadata())  # Extract metadata from context
    key_in_metadata = metadata.get('x-api-key')  # Extract API key from metadata

    if key_in_metadata != api_key:
        # If the API key doesn't match, raise an UNAUTHENTICATED error
        raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key.")
    return True

class FacialRecognitionService(facial_recognition_pb2_grpc.FacialRecognitionServiceServicer):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing FacialRecognitionService...")

        # Load configurations
        predictor_path = "./shapes_predictor_68_face_landmarks.dat"
        self.db = Database(
                        host=os.getenv("DB_HOST"),
                        database=os.getenv("DB_NAME"),
                        user=os.getenv("DB_USER"),
                        password=os.getenv("DB_PASSWORD")
                    )
        self.landmark_detector = FacialLandmarkDetector(predictor_path)
        self.knn_model = None
        self.load_and_train_knn()

    def load_and_train_knn(self):
        """Load data from the database and train the KNN model."""
        self.logger.info("Loading and training KNN model...")
        landmarks = self.db.fetch_landmarks()

        if landmarks:
            X = np.array([np.ravel(data[1]) for data in landmarks]).reshape(len(landmarks), -1)
            y = np.array([data[0] for data in landmarks])
            self.knn_model = KNeighborsClassifier(n_neighbors=1)
            self.knn_model.fit(X, y)
            self.logger.info(f"KNN model trained with {len(landmarks)} records.")
        else:
            self.knn_model = None
            self.logger.warning("No data available to train KNN model.")

    @staticmethod
    def decode_base64_image(base64_image):
        """Decode Base64-encoded image data into raw binary."""
        try:
            return np.frombuffer(base64.b64decode(base64_image), np.uint8)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 image: {e}")

    def SaveImage(self, request, context):
        """Save image landmarks for a user."""
        # Check the API key before proceeding
        try:
            check_api_key(context)
        except grpc.RpcError:
            return facial_recognition_pb2.SaveImageResponse(success=False, message="Invalid API key.")

        user_id = request.user_id
        self.logger.info(f"Processing SaveImage request for user ID: {user_id}")

        # Check if user_id already exists
        if self.db.check_user_exists(user_id):
            return facial_recognition_pb2.SaveImageResponse(success=False, message=f"User ID {user_id} already exists.")

        try:
            img_data = self.decode_base64_image(request.image)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if img is None:
                return facial_recognition_pb2.SaveImageResponse(success=False, message="Invalid image file.")

            face_landmarks = self.landmark_detector.detect_landmarks(img)
            if face_landmarks is None:
                return facial_recognition_pb2.SaveImageResponse(success=False, message="No face detected.")

            flattened_landmarks = [coord for point in face_landmarks for coord in point]

            if self.db.save_landmarks(user_id, flattened_landmarks):
                self.load_and_train_knn()
                return facial_recognition_pb2.SaveImageResponse(
                    success=True,
                    message="Landmarks saved successfully.",
                    landmarks=flattened_landmarks,
                )
            else:
                return facial_recognition_pb2.SaveImageResponse(success=False, message="Database save failed.")

        except Exception as e:
            self.logger.error(f"Error in SaveImage: {e}")
            return facial_recognition_pb2.SaveImageResponse(success=False, message=f"Internal error: {e}")

    def GetUserId(self, request, context):
        """Retrieve the user ID based on facial landmarks."""
        # Check the API key before proceeding
        try:
            check_api_key(context)
        except grpc.RpcError as e:
            return facial_recognition_pb2.GetUserIdResponse(success=False, message="Invalid API key.")

        self.logger.info("Processing GetUserId request.")

        try:
            img_data = self.decode_base64_image(request.image)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if img is None:
                return facial_recognition_pb2.GetUserIdResponse(success=False, message="Invalid image file.")

            face_landmarks = self.landmark_detector.detect_landmarks(img)
            if face_landmarks is None:
                return facial_recognition_pb2.GetUserIdResponse(success=False, message="No face detected.")

            if self.knn_model:
                flattened_landmarks = [coord for point in face_landmarks for coord in point]
                face_landmarks_np = np.array(flattened_landmarks).reshape(1, -1)
                user_id = self.knn_model.predict(face_landmarks_np)
                return facial_recognition_pb2.GetUserIdResponse(
                    success=True,
                    message="User ID retrieved successfully.",
                    user_id=int(user_id[0]),
                )
            else:
                return facial_recognition_pb2.GetUserIdResponse(success=False, message="KNN model is not trained.")

        except Exception as e:
            self.logger.error(f"Error in GetUserId: {e}")
            return facial_recognition_pb2.GetUserIdResponse(success=False, message=f"Internal error: {e}")

def serve():
    """Start the gRPC server."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gRPC Server")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    facial_recognition_pb2_grpc.add_FacialRecognitionServiceServicer_to_server(
        FacialRecognitionService(), server
    )
    port = os.getenv("GRPC_PORT", "50051")
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"gRPC Server starting on port {port}...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
