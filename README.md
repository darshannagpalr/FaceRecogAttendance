# Face Recognition System with Flask and MySQL

A web-based face recognition and attendance/login system built with Flask, OpenCV, and MySQL. This application allows you to add new faces, train a recognition model, perform real-time face recognition via webcam, and log recognized users into a database.

## Table of Contents

-   [Features](#features)
-   [Technologies Used](#technologies-used)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Database Setup](#database-setup)
-   [Running the Application](#running-the-application)
-   [Usage Guide](#usage-guide)
    -   [Adding New Faces](#adding-new-faces-add_face)
    -   [Training the Model](#training-the-model)
    -   [Starting/Stopping Face Recognition](#startingstopping-face-recognition)
    -   [Viewing Login History](#viewing-login-history)
-   [Project Structure](#project-structure)
-   [Important Notes](#important-notes)
-   [Future Enhancements](#future-enhancements)
-   [License](#license)

## Features

* **Web-based Interface:** Easy interaction via a modern web browser.
* **Add New Faces:** Capture live images from your webcam to add new individuals to the recognition dataset.
* **Model Training:** Train the face recognition model with your collected dataset.
* **Real-time Recognition:** Stream live webcam feed and identify pre-trained faces.
* **Database Logging:** Automatically logs recognized individuals (after a few continuous detections) into a MySQL database with their name and login timestamp.
* **Session-based Skipping:** Prevents re-logging the same person multiple times within a single recognition session.
* **Login History:** View a chronological list of all logged-in individuals.

## Technologies Used

* **Backend:** Python 3, Flask
* **Face Recognition:** OpenCV (with LBPHFaceRecognizer)
* **Database:** MySQL (via `mysql-connector-python`)
* **Frontend:** HTML, CSS, JavaScript (for webcam interaction)
* **Image Processing:** NumPy, Pillow

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.x**: Download from [python.org](https://www.python.org/downloads/).
* **pip**: Python package installer (usually comes with Python).
* **MySQL Server**: Install MySQL Community Server or use a local Docker container.
    * Download from [dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/).
* **Webcam**: A functional webcam connected to your system.
* **`haarcascade_frontalface_default.xml`**: This XML file is essential for face detection. It should be placed in the root directory of your project (alongside `app.py`). You can usually find it in your OpenCV installation directory (e.g., `opencv/data/haarcascades/haarcascade_frontalface_default.xml`) or download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/darshannagpalr/face-recognition-system.git](https://github.com/darshannagpalr/face-recognition-system.git)
    cd face-recognition-system
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Python dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    Flask
    mysql-connector-python
    opencv-python
    numpy
    Pillow
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Database Setup

1.  **Start your MySQL Server.**
2.  **Log in to your MySQL server** (e.g., using MySQL Workbench, a command-line client, or phpMyAdmin) as a user with sufficient privileges (e.g., `root`).
3.  **Create the database:**
    ```sql
    CREATE DATABASE faces;
    ```
4.  **Update `DB_CONFIG` in `app.py`:**
    Open `app.py` and modify the `DB_CONFIG` dictionary with your MySQL credentials:
    ```python
    DB_CONFIG = {
        "host": "localhost",
        "user": "root", # Your MySQL username
        "password": "root", # Your MySQL password
        "database": "faces",
    }
    ```
    *(**Security Note:** For production environments, avoid hardcoding credentials directly in the code. Use environment variables or a configuration file.)*

5.  The `setup_database()` function in `app.py` will automatically create the `login` table when the Flask application starts for the first time if it doesn't already exist.

## Running the Application

1.  **Set Flask Environment Variables:**
    ```bash
    export FLASK_APP=app.py
    export FLASK_DEBUG=True # For development, enable debug mode
    ```
    * **On Windows (Command Prompt):**
        ```cmd
        set FLASK_APP=app.py
        set FLASK_DEBUG=True
        ```
    * **On Windows (PowerShell):**
        ```powershell
        $env:FLASK_APP="app.py"
        $env:FLASK_DEBUG="True"
        ```

2.  **Set a Flask Secret Key:**
    In `app.py`, locate the line `app.secret_key = 'your_super_secret_key_here'` and **change `'your_super_secret_key_here'` to a long, random, and strong string.** This is crucial for session security.
    ```python
    app.secret_key = 'a_very_long_and_random_secret_key_for_security_purposes_12345'
    ```

3.  **Start the Flask application:**
    ```bash
    flask run
    ```

4.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage Guide

### Dashboard (`/`)

The main dashboard provides links to all functionalities: Add Face, Train Model, Start Recognition, and View Login History.

### Adding New Faces (`/add_face`)

1.  Navigate to the "Add Face" page from the Dashboard.
2.  Enter the **Person's Name** in the input field.
3.  Click "Start Capture". Your webcam feed should appear.
4.  Click "Take Photo" multiple times to capture images of the person's face. Aim for **10-20 images** from various angles and expressions to improve recognition accuracy. Thumbnails of captured images will appear.
5.  Click "Stop Capture" when you are done.
6.  The images will be saved automatically to `dataset/Person_Name/` (where `Person_Name` is the sanitized name you entered).

### Training the Model

After adding new faces or modifying the `dataset` directory (e.g., adding images manually), you *must* train the model:

1.  Click the "Train Model Now" button on the Dashboard.
2.  A success or error message will be displayed indicating the training status. Check your server console for detailed logs.
3.  This step creates/updates the `trainer/trainer.yml` and `trainer/labels.json` files.

### Starting/Stopping Face Recognition (`/recognition_stream`)

1.  Navigate to the "Face Recognition" page from the Dashboard.
2.  Click the "Start Recognition" button. Your webcam feed will begin streaming, and the system will attempt to identify faces.
3.  Recognized faces will have their name and confidence level displayed on the stream.
4.  If a recognized face is detected continuously for a few frames and is not already logged in the database for the current session, their entry will be added to the `login` table.
5.  Click "Stop Recognition" to end the recognition stream and release the webcam.

### Viewing Login History (`/login_history`)

1.  Navigate to the "Login History" page from the Dashboard.
2.  This page displays a table of all individuals logged into the system, including their ID, Name, and Login Time.

## Project Structure

```sh
└── FaceRecogAttendance/
    ├── app.py
    ├── haarcascade_frontalface_default.xml
    ├── static
    │   ├── css
    │   └── js
    ├── templates
    │   ├── add_face.html
    │   ├── base.html
    │   ├── index.html
    │   ├── login_history.html
    │   └── recognition_stream.html
    └── trainer
        ├── labels.json
        └── trainer.yml
```

## Important Notes

* **Security:** This is a basic demonstration. For production environments, consider robust security measures, including secure credential management, proper authentication, and error handling.
* **Accuracy:** Face recognition accuracy depends heavily on the quality and quantity of training data. Capture clear images with varied lighting, expressions, and angles.
* **Performance:** The recognition performance can vary based on your system's hardware and webcam quality.
* **Haar Cascade:** Ensure the `haarcascade_frontalface_default.xml` file is in the root directory of your project.

## Future Enhancements

* **Automated Image Capture:** Implement a feature to automatically capture a set number of images for a new person (e.g., "capture 20 images in 5 seconds").
* **Real-time Feedback during Capture:** Provide real-time feedback (e.g., "face detected", "image saved") on the `add_face` page.
* **User Management:** Implement user authentication for accessing the web interface.
* **Advanced Recognition Models:** Integrate more advanced deep learning models (e.g., FaceNet, ArcFace) for higher accuracy.
* **Logging Out:** Add functionality to mark users as "logged out" from the database.
* **Attendance Reports:** Generate reports based on login history.
* **Dockerization:** Provide Dockerfiles for easy deployment.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.
