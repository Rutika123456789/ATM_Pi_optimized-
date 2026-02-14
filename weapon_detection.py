import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

# =============================
# CONFIG
# =============================
MODEL_PATH = "best_float32.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
REQUIRED_FRAMES = 2

CLASS_NAMES = [
    "gun",
    "knife",
    "pistol"
]

WEAPON_CLASSES = set(CLASS_NAMES)

# =============================
# INIT MODEL
# =============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =============================
# INIT CAMERA
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =============================
# STORAGE
# =============================
os.makedirs("alerts", exist_ok=True)
weapon_count = 0

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # -------- PREPROCESS --------
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # -------- INFERENCE --------
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]  # (6, 8400)

    detected = False
    weapon_type = None

    # -------- PARSE OUTPUT --------
    for i in range(output.shape[1]):
        conf = output[4, i]
        cls_id = int(output[5, i])

        if conf < CONF_THRESHOLD:
            continue

        if cls_id >= len(CLASS_NAMES):
            continue

        label = CLASS_NAMES[cls_id]

        if label in WEAPON_CLASSES:
            detected = True
            weapon_type = label
            break

    # -------- TEMPORAL STABILITY --------
    if detected:
        weapon_count += 1
    else:
        weapon_count = max(0, weapon_count - 1)

    # -------- ALERT --------
    if weapon_count >= REQUIRED_FRAMES:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        alert_path = f"alerts/weapon_{weapon_type}_{timestamp}.jpg"
        cv2.imwrite(alert_path, frame)
        print(f"🚨 WEAPON DETECTED: {weapon_type.upper()}")
        break

    # -------- DISPLAY --------
    cv2.imshow("ATM Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
