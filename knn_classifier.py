import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from db_connect import Database

class KNNModel:
    def __init__(self, db: Database, n_neighbors=1):
        self.db = db  # Database connection
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X = []  # Feature matrix (landmarks)
        self.y = []  # Labels (user IDs)

    def load_data(self):
        """Load landmarks and user IDs from the database."""
        landmarks = self.db.fetch_landmarks()  # Assuming you have a method to fetch landmarks
        if landmarks:
            self.X = [np.array(data[1]) for data in landmarks]  # Access the second element of the tuple (landmarks)
            self.y = [data[0] for data in landmarks]  # Assuming first element is user_id
            return True
        return False

    def fit(self):
        """Train the KNN model on the loaded data."""
        if self.X and self.y:
            # Reshape self.X to flatten the 2D landmarks into 1D vectors
            self.X = np.array(self.X)  # Convert to numpy array if it's not already
            self.X = self.X.reshape(self.X.shape[0], -1)  # Flatten (n_samples, n_points * 2)

            self.knn.fit(self.X, self.y)  # Fit the model
            print("KNN model trained successfully.")
        else:
            print("No data to train the KNN model.")

    def add_new_data(self, user_id, new_landmarks):
        """Add new landmarks and retrain the model."""
        self.X.append(np.array(new_landmarks))  # Append new landmarks
        self.y.append(user_id)  # Append corresponding user ID
        self.fit()  # Retrain the model with the updated dataset

    def predict(self, new_landmarks):
        """Predict the user ID for the given new landmarks."""
        if self.knn:
            new_landmarks = np.array(new_landmarks).reshape(1, -1)  # Flatten new landmarks for prediction
            prediction = self.knn.predict(new_landmarks)[0]  # Predict for the new landmarks
            return prediction
        return None
