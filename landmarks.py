import cv2
import dlib
import numpy as np


class FacialLandmarkDetector:
    def __init__(self, predictor_path="./shapes_predictor_68_face_landmarks.dat"):
        """
        Initializes the FacialLandmarkDetector with the given predictor path.

        :param predictor_path: Path to the dlib shape predictor model file.
        """
        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()

        # Initialize dlib's shape predictor
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading shape predictor: {e}. "
                               f"Ensure the file '{predictor_path}' exists and the path is correct.")

    def detect_landmarks(self, img):
        """
        Detects facial landmarks in an image.

        :param img: Image as a numpy array (BGR format).
        :return: List of landmark coordinates for the first detected face, or None if no face is detected.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray)

        if len(faces) == 0:
            return None  # No faces detected

        # Assuming only one face per image; process the first detected face
        face = faces[0]
        shape = self.predictor(gray, face)

        # Extract the (x, y) coordinates of the 68 landmarks
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        return landmarks
