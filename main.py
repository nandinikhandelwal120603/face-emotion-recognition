import tensorflow
from tensorflow import keras
from keras.models import load_model
from time import time, sleep
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the face classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier(r'C:\Users\khand\OneDrive\Documents\python\ai face and emotion recognition\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\khand\OneDrive\Documents\python\ai face and emotion recognition\model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)

start_time = time()  # Start time
last_emotion = None  # Track last detected non-neutral emotion

while time() - start_time < 10:  # Run for 10 seconds
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)

            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Store last detected non-neutral emotion
            if label != "Neutral":
                last_emotion = label
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the last detected emotion (not Neutral)
if last_emotion:
    print(f"Final detected emotion (excluding Neutral): {last_emotion}")
else:
    print("No strong emotion detected.")
