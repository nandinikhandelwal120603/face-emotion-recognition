'''
How It Works Now:
Face images are captured and the model is trained.
The script tracks faces for the specified duration.
After the face tracking is complete, it asks if the user wants to delete their data.
If the user chooses to delete, the captured images and name mapping are removed.
If the user keeps the data, it is retained for future use.

'''

import cv2
import numpy as np
from PIL import Image
import os
import json
import logging
from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS
from time import sleep



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def get_face_id(directory: str) -> int:
    try:
        if not os.path.exists(directory):
            return 1
            
        user_ids = []
        for filename in os.listdir(directory):
            if filename.startswith('Users-'):
                try:
                    number = int(filename.split('-')[1])
                    user_ids.append(number)
                except (IndexError, ValueError):
                    continue
        return max(user_ids + [0]) + 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        raise

def save_name(face_id: int, face_name: str, filename: str) -> None:
    try:
        names_json = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as fs:
                    content = fs.read().strip()
                    if content:
                        names_json = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {filename}, starting fresh")
                names_json = {}
        
        names_json[str(face_id)] = face_name
        
        with open(filename, 'w') as fs:
            json.dump(names_json, fs, indent=4, ensure_ascii=False)
        logger.info(f"Saved name mapping for ID {face_id}")
    except Exception as e:
        logger.error(f"Error saving name mapping: {e}")
        raise

def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            raise ValueError("Failed to initialize camera")
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        raise

def get_images_and_labels(path: str):
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        if detector.empty():
            raise ValueError("Error loading cascade classifier")

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split("-")[1].split(".")[0])

            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

def train_model():
    try:
        logger.info("Starting face recognition training...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found")
            
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

import time

def track_faces():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(PATHS['trainer_file'])
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])

        cam = initialize_camera(CAMERA['index'])

        start_time = time.time()  # Record start time
        duration = 5  # Duration in seconds

        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                with open(PATHS['names_file'], 'r') as f:
                    names_json = json.load(f)
                    name = names_json.get(str(id), "Unknown")

                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
            cv2.imshow('Face Recognition', img)

            # Break the loop if the time exceeds the limit
            if time.time() - start_time > duration:
                logger.info(f"Stopping face tracking after {duration} seconds.")
                break

            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
                break
        
        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An error occurred during face tracking: {e}")
        raise


def delete_user_data(face_id: int, image_dir: str, names_file: str) -> None:
    """
    Delete user data (images and name mapping) from the system.

    Parameters:
        face_id (int): The identifier of the user whose data is to be deleted.
        image_dir (str): The directory containing the user's images.
        names_file (str): Path to the JSON file where names and IDs are stored.
    """
    try:
        # Delete images of the user
        for filename in os.listdir(image_dir):
            if filename.startswith(f'Users-{face_id}-'):
                file_path = os.path.join(image_dir, filename)
                os.remove(file_path)
                logger.info(f"Deleted image: {file_path}")

        # Delete name mapping from JSON file
        if os.path.exists(names_file):
            with open(names_file, 'r') as fs:
                names_json = json.load(fs)

            if str(face_id) in names_json:
                del names_json[str(face_id)]
                with open(names_file, 'w') as fs:
                    json.dump(names_json, fs, indent=4, ensure_ascii=False)
                logger.info(f"Deleted name mapping for ID {face_id}")
            else:
                logger.warning(f"No name mapping found for ID {face_id}")
        
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        raise


if __name__ == '__main__':
    try:
        # Step 1: Face Taker (Capture images)
        create_directory(PATHS['image_dir'])
        face_name = input("Enter user name: ").strip()
        face_id = get_face_id(PATHS['image_dir'])
        save_name(face_id, face_name, PATHS['names_file'])
        
        cam = initialize_camera(CAMERA['index'])
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])

        count = 0
        while count < TRAINING['samples_needed']:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            for (x, y, w, h) in faces:
                count += 1
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                img_path = f"{PATHS['image_dir']}/Users-{face_id}-{count}.jpg"
                cv2.imwrite(img_path, face_img)

            cv2.imshow('Capturing', img)
            if cv2.waitKey(100) & 0xFF == 27:
                break
        
        cam.release()
        cv2.destroyAllWindows()

        # Step 2: Train the model
        train_model()

        # Step 3: Track faces
        track_faces()

        # Step 4: Ask if the user wants to delete the captured data
        delete_choice = input(f"Do you want to delete the captured data for {face_name} (ID: {face_id})? (yes/no): ").strip().lower()
        if delete_choice == 'yes':
            delete_user_data(face_id, PATHS['image_dir'], PATHS['names_file'])
            logger.info(f"User {face_name} (ID: {face_id}) data deleted.")
        else:
            logger.info(f"User {face_name} (ID: {face_id}) data kept.")
        
    except Exception as e:
        logger.error(f"An error occurred in the combined workflow: {e}")
