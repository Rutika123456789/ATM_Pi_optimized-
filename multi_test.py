import cv2
import numpy as np
import json
import subprocess
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
    '04 - 06 years old', '07 - 11 years old', '12 - 17 years old',
    '18 - 23 years old', '24 - 27 years old', '28 - 35 years old',
    '36 - 45 years old', '46 - 60 years old', '61 - 75 years old'
]

# ==============================
# LOAD MODELS
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
# CLASS INDICES
# ==============================
try:
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    class_indices = {'with_mask': 0, 'without_mask': 1}

mask_index = class_indices.get('with_mask', 0)

# ==============================
# FACE DETECTOR
# ==============================
faceNet = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# ==============================
# TTS (FESTIVAL FEMALE VOICE)
# ==============================
last_tts_time = 0
TTS_COOLDOWN = 3

def festival_tts(text):
    scheme = f'(voice_cmu_us_slt_arctic_hts) (SayText "{text}")'
    subprocess.run(["festival"], input=scheme, text=True)

def tts_guard(msg):
    global last_tts_time
    if time() - last_tts_time > TTS_COOLDOWN:
        festival_tts(msg)
        last_tts_time = time()

# ==============================
# FACE DETECTION (ALL FACES)
# ==============================
def detect_faces(frame, threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104, 177, 123)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) > 50 and (y2 - y1) > 50:
                faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces

# ==============================
# 🔐 MAIN SECURITY FUNCTION
# ==============================
def run_security_check(timeout=20):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "DENIED"

    start_time = time()

    FACE_COUNT_BUFFER = []
    BUFFER_SIZE = 20
    locked_face = None

    while time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        if locked_face is None:
            faces = detect_faces(frame)

            FACE_COUNT_BUFFER.append(len(faces))
            if len(FACE_COUNT_BUFFER) > BUFFER_SIZE:
                FACE_COUNT_BUFFER.pop(0)

            stable_count = max(set(FACE_COUNT_BUFFER), key=FACE_COUNT_BUFFER.count)

            if stable_count == 0:
                continue

            if stable_count > 1:
                tts_guard("Multiple people detected. Access denied.")
                FACE_COUNT_BUFFER.clear()
                continue

            if FACE_COUNT_BUFFER.count(1) < 15:
                continue

            x, y, w, h = faces[0]
            locked_face = frame[y:y+h, x:x+w].copy()
            continue

        face = locked_face
        if face.size == 0:
            continue

        mask_img = cv2.resize(face, IMG_SIZE_MASK)
        mask_img = mask_img.astype(np.float32) / 255.0
        mask_img = np.expand_dims(mask_img, axis=0)

        interpreter_mask.set_tensor(mask_input[0]['index'], mask_img)
        interpreter_mask.invoke()
        mask_score = interpreter_mask.get_tensor(mask_output[0]['index'])[0][0]

        is_mask = mask_score > MASK_THRESHOLD if mask_index == 1 else mask_score < MASK_THRESHOLD

        if is_mask:
            tts_guard("Please remove your mask and stay still.")
            FACE_COUNT_BUFFER.clear()
            locked_face = None
            continue

        age_img = cv2.resize(face, AGE_INPUT_SIZE)
        age_img = age_img.astype(np.float32) / 255.0
        age_img = np.expand_dims(age_img, axis=0)

        interpreter_age.set_tensor(age_input[0]['index'], age_img)
        interpreter_age.invoke()
        preds = interpreter_age.get_tensor(age_output[0]['index'])
        age_idx = int(np.argmax(preds))

        if age_idx >= 3:
            tts_guard("Access granted. Please proceed.")
            cap.release()
            cv2.destroyAllWindows()
            return "GRANTED"
        else:
            tts_guard("You are under age. Access denied.")
            cap.release()
            cv2.destroyAllWindows()
            return "DENIED"

    cap.release()
    cv2.destroyAllWindows()
    return "DENIED"
