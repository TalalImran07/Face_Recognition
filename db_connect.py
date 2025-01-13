import psycopg2
from psycopg2.extras import Json
import os
import json


class Database:
    def __init__(self, host, database, user, password):
        # Fetch database connection details from environment variables
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def get_connection(self):
        """Establishes and returns a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return conn
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

    def save_landmarks(self, user_id, landmarks):
        """Saves the user ID and landmarks into the database."""
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cur = conn.cursor()
            # Insert user_id and landmarks as JSON
            cur.execute(
                "INSERT INTO landmarks (user_id, landmarks) VALUES (%s, %s)",
                (user_id, json.dumps(landmarks))  # Convert landmarks to JSON format
            )
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            print(f"Error saving landmarks to database: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def fetch_landmarks(self):
        """Fetches all landmarks from the database."""
        conn = self.get_connection()
        if conn is None:
            return []

        landmarks_data = []
        try:
            cur = conn.cursor()
            cur.execute("SELECT user_id, landmarks FROM landmarks")
            rows = cur.fetchall()
            for row in rows:
                user_id, landmarks = row
                # Convert JSON string to Python list if necessary
                if isinstance(landmarks, str):
                    landmarks_data.append((user_id, json.loads(landmarks)))
                elif isinstance(landmarks, list):
                    landmarks_data.append((user_id, landmarks))
                else:
                    print("Unexpected format for landmarks. Must be a string or list.")
            cur.close()
        except Exception as e:
            print(f"Error fetching landmarks from database: {e}")
        finally:
            if conn:
                conn.close()

        return landmarks_data

    def check_user_exists(self, user_id):
        """Check if a user ID already exists in the database."""
        conn = self.get_connection()
        if conn is None:
            return False

        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM landmarks WHERE user_id = %s LIMIT 1", (user_id,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
        except Exception as e:
            print(f"Error checking if user ID exists: {e}")
            return False
        finally:
            if conn:
                conn.close()
