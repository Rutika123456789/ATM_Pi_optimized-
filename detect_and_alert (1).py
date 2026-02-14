import cv2
import numpy as np
import pyttsx3
import time
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#Configuration
MODEL_PATH = "mask_detection/models/best_mask_detector.h5"
CLASS_INDEX_PATH = "mask_detection/models/class_indices.json"  # saved after training
IMG_SIZE = (150, 150)
MASK_THRESHOLD = 0.5

#Load Mask Model
maskNet = load_model(MODEL_PATH)

#Load Class Indices
try:
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    print("[INFO] Loaded class indices:", class_indices)
except FileNotFoundError:
    print("[WARNING] class_indices.json not found. Defaulting to {'with_mask': 0, 'without_mask': 1}")
    class_indices = {'with_mask': 0, 'without_mask': 1}

# Detect which index corresponds to "with_mask"
mask_index = class_indices.get('with_mask', 0)

#Load DNN Face Detector
prototxt_path = "deploy.prototxt"
weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

#Helper: Detect faces
def detect_faces_dnn(frame, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

#Video Stream 
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Cannot access camera")
    exit()

tts_count = 0
last_tts_time = 0
tts_limit = 3
tts_cooldown = 3  # seconds between TTS

print("[INFO] Starting Real-Time Mask Detection (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces_dnn(frame)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, IMG_SIZE)
            face_arr = img_to_array(face_resized).astype("float") / 255.0
            face_arr = np.expand_dims(face_arr, axis=0)

            pred = maskNet.predict(face_arr, verbose=0)[0][0]

            # === Automatic label based on class_indices ===
            if mask_index == 1:
                is_mask = pred > MASK_THRESHOLD
            else:
                is_mask = pred < MASK_THRESHOLD

            if is_mask:
                label = "With Mask"
                color = (0, 0, 255)  # Red
                if tts_count < tts_limit and (time.time() - last_tts_time > tts_cooldown):
                    speak("Please uncover your face.")
                    tts_count += 1
                    last_tts_time = time.time()
            else:
                label = "Without Mask"
                color = (0, 255, 0)  # Green
                tts_count = 0  # reset when no mask

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    else:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame,
                "For identity verification,Look straight at the camera",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Real-Time Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speed of speech

# Load the trained model
model = tf.keras.models.load_model('mask_detection/models/mask_detector.h5')

# Labels
labels = ['Mask', 'No Mask']

# Initialize webcam
cap = cv2.VideoCapture(0)

# To avoid constant repetition, use a flag
mask_spoken = False
nomask_spoken = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = labels[class_index]

    # Show result on the frame
    color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
    cv2.putText(frame, label + " Detected", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (30, 30), (600, 400), color, 2)

    # Speak result continuously if mask detected
    if label == 'Mask':
        if not mask_spoken:
            engine.say("Mask detected")
            engine.runAndWait()
            mask_spoken = True
            nomask_spoken = False
    else:
        if not nomask_spoken:
            engine.say("No mask detected")
            engine.runAndWait()
            nomask_spoken = True
            mask_spoken = False

    # Show the webcam window
    cv2.imshow("Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""