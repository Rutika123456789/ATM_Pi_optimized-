import cv2
import numpy as np
import threading
import time
import os
from tflite_runtime.interpreter import Interpreter

# ===============================
# MODEL PATHS (YOUR FILES)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AGE_MODEL_PATH = os.path.join(BASE_DIR, "AgeClass_best_06_02-16-02.tflite")
MASK_MODEL_PATH = os.path.join(BASE_DIR, "best_mask_detector_float32.tflite")
WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "best_float32.tflite")

# ===============================
# LOAD INTERPRETERS
# ===============================
age_interpreter = Interpreter(model_path=AGE_MODEL_PATH)
mask_interpreter = Interpreter(model_path=MASK_MODEL_PATH)
weapon_interpreter = Interpreter(model_path=WEAPON_MODEL_PATH)

age_interpreter.allocate_tensors()
mask_interpreter.allocate_tensors()
weapon_interpreter.allocate_tensors()

age_input = age_interpreter.get_input_details()
mask_input = mask_interpreter.get_input_details()
weapon_input = weapon_interpreter.get_input_details()

age_output = age_interpreter.get_output_details()
mask_output = mask_interpreter.get_output_details()
weapon_output = weapon_interpreter.get_output_details()

print("Models loaded successfully")

# ===============================
# PREPROCESS FUNCTION (FLOAT32)
# ===============================
def preprocess(frame, input_details):
    h, w = input_details[0]['shape'][1:3]
    img = cv2.resize(frame, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# AGE DETECTION
# ===============================
def detect_age(frame):
    img = preprocess(frame, age_input)
    age_interpreter.set_tensor(age_input[0]['index'], img)
    age_interpreter.invoke()
    output = age_interpreter.get_tensor(age_output[0]['index'])
    return int(np.argmax(output))

# ===============================
# MASK DETECTION
# ===============================
def detect_mask(frame):
    img = preprocess(frame, mask_input)
    mask_interpreter.set_tensor(mask_input[0]['index'], img)
    mask_interpreter.invoke()
    output = mask_interpreter.get_tensor(mask_output[0]['index'])
    confidence = np.max(output)
    return confidence > 0.5

# ===============================
# WEAPON DETECTION
# ===============================
def detect_weapon(frame):
    img = preprocess(frame, weapon_input)
    weapon_interpreter.set_tensor(weapon_input[0]['index'], img)
    weapon_interpreter.invoke()
    output = weapon_interpreter.get_tensor(weapon_output[0]['index'])

    # YOLO-style assumption
    confidence = np.max(output)
    return confidence > 0.5

# ===============================
# THREAD: WEAPON MONITOR
# ===============================
def weapon_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            if detect_weapon(frame):
                print("⚠️ WEAPON DETECTED!")
        except Exception as e:
            print("Weapon thread error:", e)

        time.sleep(0.05)

# ===============================
# MAIN
# ===============================
def main():
    print("ATM Security System Started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return

    threading.Thread(target=weapon_thread, args=(cap,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        age_class = detect_age(frame)
        mask_ok = detect_mask(frame)

        if age_class >= 1 and mask_ok:
            print("RESULT: GRANTED")
        else:
            print("RESULT: DENIED")

        time.sleep(1)

# ===============================
if __name__ == "__main__":
    main()
