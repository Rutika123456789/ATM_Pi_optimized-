import cv2
import numpy as np
import json
import pyttsx3
from time import time
from tflite_runtime.interpreter import Interpreter

# ==============================
# CONFIGURATION
# ==============================
AGE_MODEL_PATH = "AgeClass_best_06_02-16-02.tflite"
MASK_MODEL_PATH = "best_mask_detector_float32.tflite"
CLASS_INDEX_PATH = "class_indices.json"

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

IMG_SIZE_MASK = (150, 150)
AGE_INPUT_SIZE = (224, 224)
MASK_THRESHOLD = 0.5

AGE_LABELS = [
    '04 - 06 years old', '07 -  11 years old', '12 - 17 years old',
    '18 - 23 years old', '24 - 27 years old', '28 - 35 years old',
    '36 - 45 years old', '46 - 60 years old', '61 - 75 years old'
]

# ==============================
# LOAD TFLITE MODELS
# ==============================
interpreter_mask = Interpreter(model_path=MASK_MODEL_PATH)
interpreter_mask.allocate_tensors()
mask_input = interpreter_mask.get_input_details()
mask_output = interpreter_mask.get_output_details()

interpreter_age = Interpreter(model_path=AGE_MODEL_PATH)
interpreter_age.allocate_tensors()
age_input = interpreter_age.get_input_details()
age_output = interpreter_age.get_output_details()

# ==============================
# LOAD CLASS INDICES
# ==============================
try:
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    class_indices = {'with_mask': 0, 'without_mask': 1}

mask_index = class_indices.get('with_mask', 0)

# ==============================
# LOAD FACE DETECTOR
# ==============================
faceNet = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# ==============================
# TEXT TO SPEECH
# ==============================
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ==============================
# FACE DETECTION FUNCTION
# ==============================
def detect_faces_dnn(frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ==============================
# CAMERA SETUP (Raspberry Pi)
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit()

print("[INFO] Unified Mask + Age Detection Started (Press 'q' to quit)")

tts_count = 0
last_tts_time = 0
tts_limit = 3
tts_cooldown = 3

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces_dnn(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # ------------------------------
        # MASK DETECTION
        # ------------------------------
        mask_img = cv2.resize(face, IMG_SIZE_MASK)
        mask_img = mask_img.astype(np.float32) / 255.0
        mask_img = np.expand_dims(mask_img, axis=0)

        interpreter_mask.set_tensor(mask_input[0]['index'], mask_img)
        interpreter_mask.invoke()
        mask_pred = interpreter_mask.get_tensor(mask_output[0]['index'])[0][0]

        if mask_index == 1:
            is_mask = mask_pred > MASK_THRESHOLD
        else:
            is_mask = mask_pred < MASK_THRESHOLD

        if is_mask:
            mask_label = "With Mask"
            color = (0, 0, 255)

            if tts_count < tts_limit and (time() - last_tts_time > tts_cooldown):
                speak("Please uncover your face for verification.")
                tts_count += 1
                last_tts_time = time()
        else:
            mask_label = "Without Mask"
            color = (0, 255, 0)
            tts_count = 0

        # ------------------------------
        # AGE DETECTION (Only if no mask)
        # ------------------------------
        if not is_mask:
            age_img = cv2.resize(face, AGE_INPUT_SIZE)
            age_img = age_img.astype(np.float32) / 255.0
            age_img = np.expand_dims(age_img, axis=0)

            interpreter_age.set_tensor(age_input[0]['index'], age_img)
            interpreter_age.invoke()
            age_preds = interpreter_age.get_tensor(age_output[0]['index'])
            age_label = AGE_LABELS[int(np.argmax(age_preds))]
        else:
            age_label = "Age: Unknown (Face Covered)"

        # ------------------------------
        # DISPLAY
        # ------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, mask_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, age_label, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Mask + Age Detection (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
