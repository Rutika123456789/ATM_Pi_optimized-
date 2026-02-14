import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

# =============================
# CONFIG
# =============================
MODEL_PATH = "best_float32.tflite"
INPUT_SIZE = 320      # or 416 depending on training
CONF_THRESHOLD = 0.5
REQUIRED_FRAMES = 2
WEAPON_CLASSES = {"knife", "gun", "pistol"}

# =============================
# INIT
# =============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

os.makedirs("alerts", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

weapon_count = 0

# =============================
# LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    outputs = interpreter.get_tensor(output_details[0]["index"])

    detected = False
    weapon_type = ""

    for det in outputs[0]:
        conf = det[4]
        cls_id = int(np.argmax(det[5:]))

        if conf > CONF_THRESHOLD:
            weapon_type = str(cls_id)  # map class id ’ label
            detected = True
            break

    if detected:
        weapon_count += 1
    else:
        weapon_count = max(0, weapon_count - 1)

    if weapon_count >= REQUIRED_FRAMES:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"alerts/weapon_{ts}.jpg", frame)
        print("=¨ WEAPON DETECTED")
        break

    cv2.imshow("ATM Feed", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
