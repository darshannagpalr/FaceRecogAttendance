# import cv2
# import os
# import numpy as np
# from PIL import Image
# import json
# import mysql.connector
# from datetime import datetime


# DATASET_PATH = "dataset"
# TRAINER_PATH = "trainer/trainer.yml"
# LABELS_FILE = "trainer/labels.json"
# FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"


# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "root",
#     "database": "faces",
# }


# os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)


# def get_db_connection():
#     """
#     Establishes and returns a connection to the MySQL database.
#     """
#     conn = None
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         print(f"[INFO] Connected to MySQL database '{DB_CONFIG['database']}'.")
#         return conn
#     except mysql.connector.Error as err:
#         if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
#             print("[ERROR] Something is wrong with your user name or password")
#         elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
#             print("[ERROR] Database does not exist. Please create it first.")
#         else:
#             print(f"[ERROR] MySQL connection error: {err}")
#         return None


# def setup_database():
#     """
#     Connects to the MySQL database and creates the 'login' table if it doesn't exist.
#     The table stores recognized person's ID, name, and login time.
#     """
#     conn = get_db_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute(
#                 """
#                 CREATE TABLE IF NOT EXISTS login (
#                     id VARCHAR(255) PRIMARY KEY,
#                     name VARCHAR(255) NOT NULL,
#                     login_time DATETIME NOT NULL
#                 )
#             """
#             )
#             conn.commit()
#             print(f"[INFO] Table 'login' ensured in MySQL database.")
#         except mysql.connector.Error as err:
#             print(f"[ERROR] MySQL table creation error: {err}")
#         finally:
#             conn.close()


# def log_person_in_db(person_id, person_name):
#     """
#     Logs a person into the database if they are not already logged in.
#     Returns True if the person was logged in, False otherwise (e.g., already logged in or error).
#     """
#     conn = get_db_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()

#             cursor.execute("SELECT id FROM login WHERE id = %s", (person_id,))
#             existing_entry = cursor.fetchone()

#             if existing_entry:
#                 print(
#                     f"[INFO] Person '{person_name}' (ID: {person_id}) is already logged in. Skipping database update."
#                 )
#                 return False

#             login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             cursor.execute(
#                 "INSERT INTO login (id, name, login_time) VALUES (%s, %s, %s)",
#                 (person_id, person_name, login_time),
#             )
#             conn.commit()
#             print(f"[INFO] Logged in '{person_name}' (ID: {person_id}) at {login_time}")
#             return True
#         except mysql.connector.Error as err:
#             print(f"[ERROR] MySQL operation error for ID {person_id}: {err}")
#             return False
#         finally:
#             conn.close()
#     return False


# def get_images_and_labels(path):
#     """
#     Collects face images and their corresponding labels from the dataset directory.
#     Each sub-directory in 'path' is treated as a unique person (label).
#     Returns face samples, integer IDs, and a dictionary mapping IDs to names.
#     """

#     if not os.path.exists(path):
#         print(
#             f"[ERROR] Dataset path '{path}' does not exist. Please create it and add your images."
#         )
#         return [], [], {}

#     image_paths = []
#     for root, dirs, files in os.walk(path):
#         for dir_name in dirs:
#             person_folder = os.path.join(root, dir_name)
#             for filename in os.listdir(person_folder):
#                 if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
#                     image_paths.append(os.path.join(person_folder, filename))

#     face_samples = []
#     ids = []

#     id_to_name_map = {}
#     face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

#     if not os.path.exists(FACE_DETECTOR_PATH):
#         print(
#             f"[ERROR] Haar Cascade file '{FACE_DETECTOR_PATH}' not found. Please ensure it's in the same directory or provide the full path."
#         )
#         return [], [], {}

#     print(f"Loading images from: {path}")
#     processed_persons = set()

#     for file_full_path in image_paths:
#         try:

#             person_name = os.path.basename(os.path.dirname(file_full_path))

#             person_id = hash(person_name) % 100000

#             if person_name not in processed_persons:
#                 print(f"Processing images for: {person_name} (ID: {person_id})")
#                 processed_persons.add(person_name)

#             id_to_name_map[str(person_id)] = person_name

#             PIL_image = Image.open(file_full_path).convert("L")
#             img_numpy = np.array(PIL_image, "uint8")

#             faces = face_detector.detectMultiScale(img_numpy)

#             for x, y, w, h in faces:
#                 face_samples.append(img_numpy[y : y + h, x : x + w])
#                 ids.append(person_id)
#         except Exception as e:
#             print(f"Error processing image {file_full_path}: {e}")
#     return face_samples, ids, id_to_name_map


# def train_recognizer():
#     """
#     Trains the LBPH (Local Binary Patterns Histograms) face recognizer
#     and saves the trained model and the ID-to-name mapping.
#     Returns True if training is successful, False otherwise.
#     """
#     print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
#     faces, ids, id_to_name_map = get_images_and_labels(DATASET_PATH)

#     if not faces:
#         print(
#             "[ERROR] No faces found in the dataset or an issue occurred during image loading. "
#             "Please ensure your 'dataset' directory is correctly structured and contains valid images."
#         )
#         print("Expected structure: dataset/person_name/image.jpg")
#         return False

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, np.array(ids))

#     try:

#         recognizer.write(TRAINER_PATH)
#         print(f"\n[INFO] Model trained and saved to {TRAINER_PATH}")

#         with open(LABELS_FILE, "w") as f:
#             json.dump(id_to_name_map, f)
#         print(f"[INFO] ID-to-name mapping saved to {LABELS_FILE}")

#         print(f"[INFO] {len(np.unique(ids))} unique faces trained.")
#         return True
#     except Exception as e:
#         print(f"[ERROR] Failed to save trainer file or labels file: {e}")
#         return False


# def recognize_faces():
#     """
#     Performs real-time face recognition using the trained model via webcam.
#     """

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     try:
#         recognizer.read(TRAINER_PATH)
#     except cv2.error as e:
#         print(
#             f"[ERROR] Could not load trainer file: {TRAINER_PATH}. Please ensure you have run the training first."
#         )
#         print(f"Error details: {e}")
#         return

#     id_to_name_map = {}
#     try:
#         with open(LABELS_FILE, "r") as f:
#             id_to_name_map = json.load(f)
#     except FileNotFoundError:
#         print(
#             f"[ERROR] Labels file '{LABELS_FILE}' not found. Please ensure training was successful."
#         )
#         return
#     except json.JSONDecodeError:
#         print(
#             f"[ERROR] Could not decode labels file '{LABELS_FILE}'. It might be corrupted."
#         )
#         return

#     face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
#     if not os.path.exists(FACE_DETECTOR_PATH):
#         print(
#             f"[ERROR] Haar Cascade file '{FACE_DETECTOR_PATH}' not found. Please ensure it's in the same directory or provide the full path."
#         )
#         return

#     cam = cv2.VideoCapture(0)
#     if not cam.isOpened():
#         print(
#             "[ERROR] Could not open webcam. Please check if it's connected and not in use."
#         )
#         return

#     cam.set(3, 640)
#     cam.set(4, 480)

#     min_w = 0.1 * cam.get(3)
#     min_h = 0.1 * cam.get(4)

#     print("\n[INFO] Initializing Real-time Face Recognition. Press 'ESC' to exit.")

#     continuous_detection_counts = {}
#     logged_in_users = set()

#     while True:
#         ret, img = cam.read()
#         if not ret:
#             print("[ERROR] Failed to grab frame from camera. Exiting...")
#             break

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_detector.detectMultiScale(
#             gray,
#             scaleFactor=1.2,
#             minNeighbors=5,
#             minSize=(int(min_w), int(min_h)),
#         )

#         current_frame_recognized_ids = set()

#         for x, y, w, h in faces:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             id_int, confidence = recognizer.predict(gray[y : y + h, x : x + w])
#             id_str = str(id_int)

#             if confidence < 100:

#                 name = id_to_name_map.get(id_str, "unknown")
#                 confidence_text = f"  {round(100 - confidence)}%"

#                 current_frame_recognized_ids.add(id_str)

#                 continuous_detection_counts[id_str] = (
#                     continuous_detection_counts.get(id_str, 0) + 1
#                 )

#                 if (
#                     continuous_detection_counts[id_str] >= 3
#                     and id_str not in logged_in_users
#                 ):
#                     print(
#                         f"[INFO] {name} (ID: {id_str}) detected 3 times continuously. Attempting login."
#                     )
#                     if log_person_in_db(id_str, name):
#                         logged_in_users.add(id_str)
#                         print(f"[INFO] {name} logged in successfully.")
#                     else:
#                         print(
#                             f"[INFO] {name} was already logged in or an error occurred during login."
#                         )

#                     continuous_detection_counts[id_str] = 0

#                 cv2.putText(
#                     img,
#                     name,
#                     (x + 5, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 255, 255),
#                     2,
#                 )
#                 cv2.putText(
#                     img,
#                     confidence_text,
#                     (x + 5, y + h - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 255, 0),
#                     1,
#                 )
#             else:

#                 cv2.putText(
#                     img,
#                     "unknown",
#                     (x + 5, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 255, 255),
#                     2,
#                 )
#                 cv2.putText(
#                     img,
#                     confidence_text,
#                     (x + 5, y + h - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (255, 255, 0),
#                     1,
#                 )

#         ids_to_decrement = [
#             id_key
#             for id_key in continuous_detection_counts
#             if id_key not in current_frame_recognized_ids
#         ]
#         for id_key in ids_to_decrement:
#             continuous_detection_counts[id_key] -= 1
#             if continuous_detection_counts[id_key] <= 0:
#                 del continuous_detection_counts[id_key]

#         if len(faces) == 0:
#             continuous_detection_counts.clear()

#         cv2.imshow("Face Recognition", img)

#         k = cv2.waitKey(10) & 0xFF
#         if k == 27:
#             break

#     print("\n[INFO] Exiting Program.")
#     cam.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":

#     setup_database()

#     # train_recognizer()

#     recognize_faces()

from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import mysql.connector
import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import threading
import time

# --- Constants ---
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer/trainer.yml"
LABELS_FILE = "trainer/labels.json"
FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "faces",
}

# Ensure directories exist
os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

app = Flask(__name__)


# --- Database Functions ---
def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print(f"[INFO] Connected to MySQL database '{DB_CONFIG['database']}'.")
        return conn
    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL connection error: {err}")
        return None


def setup_database():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS login (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    login_time DATETIME NOT NULL
                )
                """
            )
            conn.commit()
            print(f"[INFO] Table 'login' ensured in MySQL database.")
        except mysql.connector.Error as err:
            print(f"[ERROR] MySQL table creation error: {err}")
        finally:
            if conn and conn.is_connected():
                conn.close()


def log_person_in_db(person_id, person_name):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Check if person_id already exists in the login table
            cursor.execute("SELECT id FROM login WHERE id = %s", (person_id,))
            existing_entry = cursor.fetchone()
            if existing_entry:
                # Removed: print(f"[INFO] Person '{person_name}' (ID: {person_id}) is already logged in the database. Skipping update.")
                return False  # Indicate that it was not a new successful login
            
            login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO login (id, name, login_time) VALUES (%s, %s, %s)",
                (person_id, person_name, login_time),
            )
            conn.commit()
            print(f"[INFO] Logged in '{person_name}' (ID: {person_id}) at {login_time}")
            return True  # Indicate successful new login
        except mysql.connector.Error as err:
            print(f"[ERROR] MySQL operation error for ID {person_id}: {err}")
            return False
        finally:
            if conn and conn.is_connected():
                conn.close()
    return False


# --- Face Recognition Core Functions ---
def get_images_and_labels(path):
    if not os.path.exists(path):
        print(
            f"[ERROR] Dataset path '{path}' does not exist. Please create it and add your images."
        )
        return [], [], {}

    image_paths = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            person_folder = os.path.join(root, dir_name)
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    image_paths.append(os.path.join(person_folder, filename))

    face_samples = []
    ids = []
    id_to_name_map = {}
    face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

    if not os.path.exists(FACE_DETECTOR_PATH):
        print(
            f"[ERROR] Haar Cascade file '{FACE_DETECTOR_PATH}' not found. Please ensure it's in the same directory or provide the full path."
        )
        return [], [], {}

    print(f"Loading images from: {path}")
    processed_persons = set()

    for file_full_path in image_paths:
        try:
            person_name = os.path.basename(os.path.dirname(file_full_path))
            person_id = hash(person_name) % 100000

            if person_name not in processed_persons:
                print(f"Processing images for: {person_name} (ID: {person_id})")
                processed_persons.add(person_name)

            id_to_name_map[str(person_id)] = person_name

            PIL_image = Image.open(file_full_path).convert("L")
            img_numpy = np.array(PIL_image, "uint8")

            faces = face_detector.detectMultiScale(img_numpy)

            for x, y, w, h in faces:
                face_samples.append(img_numpy[y : y + h, x : x + w])
                ids.append(person_id)
        except Exception as e:
            print(f"Error processing image {file_full_path}: {e}")
    return face_samples, ids, id_to_name_map


def train_recognizer():
    print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids, id_to_name_map = get_images_and_labels(DATASET_PATH)

    if not faces:
        print(
            "[ERROR] No faces found in the dataset or an issue occurred during image loading. "
        )
        print("Expected structure: dataset/person_name/image.jpg")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    try:
        recognizer.write(TRAINER_PATH)
        print(f"\n[INFO] Model trained and saved to {TRAINER_PATH}")

        with open(LABELS_FILE, "w") as f:
            json.dump(id_to_name_map, f)
        print(f"[INFO] ID-to-name mapping saved to {LABELS_FILE}")

        print(f"[INFO] {len(np.unique(ids))} unique faces trained.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save trainer file or labels file: {e}")
        return False


# Global variables for recognition thread management
recognition_thread = None
recognition_running = False
camera = None  # Global camera object
current_session_logged_in_users = set() # To track users logged in during the current active recognition session


def generate_frames():
    global recognition_running, camera, current_session_logged_in_users 

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(TRAINER_PATH)
    except cv2.error as e:
        print(
            f"[ERROR] Could not load trainer file: {TRAINER_PATH}. Recognition stopped."
        )
        recognition_running = False
        return

    id_to_name_map = {}
    try:
        with open(LABELS_FILE, "r") as f:
            id_to_name_map = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Labels file '{LABELS_FILE}' not found. Recognition stopped.")
        recognition_running = False
        return
    except json.JSONDecodeError:
        print(
            f"[ERROR] Could not decode labels file '{LABELS_FILE}'. Recognition stopped."
        )
        recognition_running = False
        return

    face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    if not os.path.exists(FACE_DETECTOR_PATH):
        print(
            f"[ERROR] Haar Cascade file '{FACE_DETECTOR_PATH}' not found. Recognition stopped."
        )
        recognition_running = False
        return

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("[ERROR] Could not open webcam. Recognition stopped.")
            recognition_running = False
            return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    min_w = 0.1 * camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    min_h = 0.1 * camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    continuous_detection_counts = {}

    while recognition_running:
        ret, img = camera.read()
        if not ret:
            print(
                "[ERROR] Failed to grab frame from camera. Recognition stream ending."
            )
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )

        current_frame_recognized_ids = set()

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id_int, confidence = recognizer.predict(gray[y : y + h, x : x + w])
            id_str = str(id_int)

            if confidence < 100:
                name = id_to_name_map.get(id_str, "unknown")
                confidence_text = f" {round(100 - confidence)}%"

                # --- NEW LOGIC: Check if already logged in for this session ---
                if id_str in current_session_logged_in_users:
                    display_text = f"{name} (Logged In)"
                    cv2.putText(
                        img,
                        display_text,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),  # Yellow color for already logged in
                        2,
                    )
                    cv2.putText(
                        img,
                        confidence_text,
                        (x + 5, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                    )
                    continue  # Skip further processing for this face if already logged in
                # --- END NEW LOGIC ---

                current_frame_recognized_ids.add(id_str)
                continuous_detection_counts[id_str] = (
                    continuous_detection_counts.get(id_str, 0) + 1
                )

                # Log person after 3 continuous detections and if not already logged in this session
                if (
                    continuous_detection_counts[id_str] >= 3
                    and id_str not in current_session_logged_in_users # Redundant check, but harmless
                ):
                    print(
                        f"[INFO] {name} (ID: {id_str}) detected 3 times continuously. Attempting login."
                    )
                    if log_person_in_db(id_str, name):
                        current_session_logged_in_users.add(id_str) # Add to the session's logged-in set
                        print(f"[INFO] {name} logged in successfully.")
                    # Removed: else block that printed "was already logged in (in DB)"
                    continuous_detection_counts[id_str] = (
                        0  # Reset count after attempting login
                    )

                cv2.putText(
                    img,
                    name,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    confidence_text,
                    (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    1,
                )
            else:
                cv2.putText(
                    img,
                    "unknown",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    " 0%",
                    (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    1,
                )

        # Decrement counts for people not detected in the current frame
        ids_to_decrement = [
            id_key
            for id_key in continuous_detection_counts
            if id_key not in current_frame_recognized_ids
        ]
        for id_key in ids_to_decrement:
            continuous_detection_counts[id_key] -= 1
            if continuous_detection_counts[id_key] <= 0:
                del continuous_detection_counts[id_key]

        # If no faces are detected in the frame, clear all continuous detection counts
        if len(faces) == 0:
            continuous_detection_counts.clear()

        ret, buffer = cv2.imencode(".jpg", img)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    if camera and camera.isOpened():
        camera.release()
        print("[INFO] Camera released.")
    cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/add_face", methods=["GET", "POST"])
def add_face():
    message = None
    if request.method == "POST":
        person_name = request.form["person_name"].strip()
        if not person_name:
            message = "Please enter a person's name."
        else:
            person_dir = os.path.join(DATASET_PATH, person_name)
            os.makedirs(person_dir, exist_ok=True)
            # In a real application, you'd integrate webcam capture or file uploads here.
            # For simplicity, we just create the directory.
            message = f"Directory for '{person_name}' created at '{person_dir}'. Please add face images there and then train the model using 'Train Model' button."
    return render_template("add_face.html", message=message)


@app.route("/train_model")
def trigger_train_model():
    if train_recognizer():
        return jsonify(
            {"status": "success", "message": "Model training initiated successfully!"}
        )
    else:
        return jsonify(
            {
                "status": "error",
                "message": "Model training failed. Check server console for details.",
            }
        )


@app.route("/login_history")
def login_history():
    conn = get_db_connection()
    login_records = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, name, login_time FROM login ORDER BY login_time DESC"
            )
            login_records = cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"[ERROR] Error fetching login history: {err}")
        finally:
            if conn and conn.is_connected():
                conn.close()
    return render_template("login_history.html", records=login_records)


@app.route("/recognition_stream")
def recognition_stream_page():
    current_year = datetime.now().year
    return render_template("recognition_stream.html", current_year=current_year)


@app.route("/start_recognition")
def start_recognition():
    global recognition_running, recognition_thread, current_session_logged_in_users 
    if not recognition_running:
        recognition_running = True
        current_session_logged_in_users.clear() # Clear the set when starting a new session
        recognition_thread = threading.Thread(target=generate_frames)
        recognition_thread.daemon = True
        recognition_thread.start()
        print("[INFO] Face recognition started.")
        return jsonify({"status": "started", "message": "Face recognition started."})
    else:
        print("[INFO] Face recognition is already running.")
        return jsonify(
            {
                "status": "already_running",
                "message": "Face recognition is already running.",
            }
        )


@app.route("/stop_recognition")
def stop_recognition():
    global recognition_running, camera
    if recognition_running:
        recognition_running = False
        if recognition_thread and recognition_thread.is_alive():
            recognition_thread.join(timeout=5)  # Give it a bit more time
        if camera and camera.isOpened():
            camera.release()
            camera = None
        print("[INFO] Face recognition stopped.")
        return jsonify({"status": "stopped", "message": "Face recognition stopped."})
    else:
        print("[INFO] Face recognition is not running.")
        return jsonify(
            {"status": "not_running", "message": "Face recognition is not running."}
        )


@app.route("/video_feed")
def video_feed():
    if recognition_running:
        return Response(
            generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    return "Recognition not running. Start it from the UI.", 400


if __name__ == "__main__":
    setup_database()  # Ensure DB table is set up on startup
    app.run(host='0.0.0.0',port=5000,debug=True)  # Run Flask in debug mode for development