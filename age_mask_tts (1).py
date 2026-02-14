import cv2
import numpy as np
import tensorflow as tf
import json
import threading
import os
from gtts import gTTS
import subprocess
import pyttsx3
from tensorflow.keras.preprocessing.image import img_to_array
from time import time

# ==============================
# CONFIGURATION
# ==============================
AGE_MODEL_PATH = "AgeClass_best_06_02-16-02.tflite"
MASK_MODEL_PATH = "best_mask_detector_float32.tflite"
CLASS_INDEX_PATH = "class_indices.json"
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

IMG_SIZE_MASK = (150, 150)
MASK_THRESHOLD = 0.5
AGE_INPUT_SIZE = (224, 224)

AGE_LABELS = [
    '04 - 06 years old', '07 - 08 years old', '09 - 11 years old',
    '12 - 19 years old', '20 - 27 years old', '28 - 35 years old',
    '36 - 45 years old', '46 - 60 years old', '61 - 75 years old'
]

# ==============================
# LOAD MODELS
# ==============================
interpreter_mask = tf.lite.Interpreter(model_path=MASK_MODEL_PATH)
interpreter_mask.allocate_tensors()
input_details_mask = interpreter_mask.get_input_details()
output_details_mask = interpreter_mask.get_output_details()

try:
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    class_indices = {'with_mask': 0, 'without_mask': 1}
mask_index = class_indices.get('with_mask', 0)

interpreter_age = tf.lite.Interpreter(model_path=AGE_MODEL_PATH)
interpreter_age.allocate_tensors()
input_details_age = interpreter_age.get_input_details()
output_details_age = interpreter_age.get_output_details()

faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

# ==============================
# TEXT TO SPEECH (Multi-engine, prioritized)
# ==============================
# Initialize pyttsx3 once for fallback
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# Attempt to select a female voice for pyttsx3 fallback
_selected_voice = None
for v in engine.getProperty('voices'):
    name_lower = (v.name or "").lower()
    id_lower = (v.id or "").lower()
    # heuristics for female voice selection
    if 'female' in name_lower or 'f' in id_lower or 'zira' in id_lower or 'sara' in name_lower or 'karen' in name_lower:
        _selected_voice = v.id
        break
# if none found, just keep default
if _selected_voice:
    engine.setProperty('voice', _selected_voice)


def _play_mp3_file(path):
    """
    Try to play an mp3 file using mpg123 (preferred) or ffplay/aplay as fallback.
    Returns True if playback succeeded, False otherwise.
    """
    # preferred: mpg123
    try:
        subprocess.run(["mpg123", "-q", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    # try ffplay (part of ffmpeg) in non-blocking mode then wait - not ideal for Pi but attempt
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    # as a last fallback try using mpg321
    try:
        subprocess.run(["mpg321", "-q", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass

    return False


def speak_text_primary_gtts(text):
    """
    Generate and play speech via gTTS with Indian variant (tld='co.in').
    Returns True if successful, False otherwise.
    """
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')  # co.in regional variant for Indian-ish voice
        tmpfile = "/tmp/tts_output.mp3"
        tts.save(tmpfile)
        played = _play_mp3_file(tmpfile)
        try:
            os.remove(tmpfile)
        except Exception:
            pass
        return played
    except Exception as e:
        print(f"[WARN] gTTS generation/playback failed: {e}")
        return False


def speak_text_espeak(text):
    """
    Use espeak (offline) with a female-sounding voice variant.
    Returns True if successful, False otherwise.
    """
    # en+f3 is commonly a female voice variant; try some options
    espeak_variants = ["en+f3", "en+f2", "en+f4", "en+f1"]
    for variant in espeak_variants:
        try:
            # use -s for speed tuning, -v for voice variant
            subprocess.run(["espeak", "-v", variant, "-s", "140", text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            continue
    # final generic espeak attempt
    try:
        subprocess.run(["espeak", "-s", "140", text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[WARN] eSpeak playback failed: {e}")
        return False


def speak_text_pyttsx3(text):
    """
    Use pyttsx3 fallback synchronous speak; returns True if no exception.
    """
    try:
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[ERROR] pyttsx3 failed: {e}")
        return False


def speak(text):
    """
    Attempt gTTS (Indian variant) -> eSpeak -> pyttsx3.
    Returns after first successful playback.
    """
    # try primary: gTTS (online)
    ok = speak_text_primary_gtts(text)
    if ok:
        return

    # fallback 1: espeak (offline)
    ok = speak_text_espeak(text)
    if ok:
        return

    # fallback 2: pyttsx3
    ok = speak_text_pyttsx3(text)
    if ok:
        return

    # If all fail, print
    print("[ERROR] All TTS engines failed to speak.")


def speak_async(text):
    """Run speak() in a daemon thread to avoid blocking the main loop."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()


# ==============================
# FACE DETECTION
# ==============================
def detect_faces_dnn(frame, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # clamp coordinates safely
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            # ensure positive width/height
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            faces.append((x1, y1, width, height))
    return faces


# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit()

print("[INFO] Starting unified detection system (Press 'q' to quit)")

last_state = None
last_change_time = 0
STATE_COOLDOWN = 2  # seconds


def build_context_message(access_state, reason=None, age_label=None):
    """
    Build a contextual speech message based on state and reason.
    """
    if access_state == "granted":
        if age_label:
            return f"Access granted. Adult verified: {age_label}."
        else:
            return "Access granted."
    if access_state == "denied":
        if reason == "multiple":
            return "Access denied — multiple people detected."
        elif reason == "mask":
            return "Please remove your face covering to verify identity."
        elif reason == "child":
            return "Access denied — only child present."
        elif reason == "no_face":
            return "No face detected. Please step in front of the camera."
        else:
            return "Access denied."
    # fallback/unknown
    return "System status unknown."


# ==============================
# MAIN LOOP
# ==============================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not captured.")
            break

        faces = detect_faces_dnn(frame)
        access_state = "unknown"
        reason = None
        age_label = None
        label_text = ""
        color = (255, 255, 0)

        # Rule 1: More than one person → access denied
        if len(faces) > 1:
            access_state = "denied"
            reason = "multiple"
            label_text = "Multiple faces detected"
            color = (0, 0, 255)

        elif len(faces) == 1:
            (x, y, w, h) = faces[0]
            # ensure we don't index out of bounds
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            face = frame[y:y2, x:x2]
            if face.size == 0:
                # weird crop; skip frame
                continue

            # ----- Mask Detection -----
            try:
                face_mask = cv2.resize(face, IMG_SIZE_MASK)
                face_mask = img_to_array(face_mask).astype("float32") / 255.0
                face_mask = np.expand_dims(face_mask, axis=0)

                interpreter_mask.set_tensor(input_details_mask[0]['index'], face_mask)
                interpreter_mask.invoke()
                # Some TFLite models output a vector of two probs, some output a single prob.
                pred_mask_raw = interpreter_mask.get_tensor(output_details_mask[0]['index'])
                # normalize extraction
                if pred_mask_raw is None:
                    pred_mask = 0.0
                else:
                    pred_mask_arr = np.array(pred_mask_raw).flatten()
                    if pred_mask_arr.size == 1:
                        pred_mask = float(pred_mask_arr[0])
                    elif pred_mask_arr.size >= 2:
                        # If mapping: [prob_with_mask, prob_without_mask] or vice versa,
                        # prefer using class_indices if available; otherwise take first element.
                        # We assume model is single-prob where higher means with_mask if mask_index==0
                        pred_mask = float(pred_mask_arr[0])
                    else:
                        pred_mask = 0.0
            except Exception as e:
                print(f"[WARN] Mask detection error: {e}")
                pred_mask = 0.0

            is_mask = (pred_mask > MASK_THRESHOLD) if mask_index == 1 else (pred_mask < MASK_THRESHOLD)

            if is_mask:
                access_state = "denied"
                reason = "mask"
                label_text = "Face Covered"
                color = (0, 0, 255)
            else:
                # ----- Age Detection -----
                try:
                    input_im = cv2.resize(face, AGE_INPUT_SIZE)
                    input_im = input_im.astype('float32') / 255.0
                    input_im = np.expand_dims(input_im, axis=0)
                    interpreter_age.set_tensor(input_details_age[0]['index'], input_im)
                    interpreter_age.invoke()
                    output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
                    output_arr = np.array(output_data_age).flatten()
                    if output_arr.size == 0:
                        index_pred_age = 0
                    else:
                        index_pred_age = int(np.argmax(output_arr))
                    index_pred_age = max(0, min(index_pred_age, len(AGE_LABELS) - 1))
                    age_label = AGE_LABELS[index_pred_age]
                except Exception as e:
                    print(f"[WARN] Age detection error: {e}")
                    index_pred_age = 0
                    age_label = AGE_LABELS[index_pred_age]

                # Adult threshold logic
                try:
                    adult_index = AGE_LABELS.index('12 - 19 years old')
                except ValueError:
                    adult_index = 3  # fallback if label not found

                if index_pred_age >= adult_index:
                    access_state = "granted"
                    label_text = age_label
                    color = (0, 255, 0)
                else:
                    access_state = "denied"
                    reason = "child"
                    label_text = age_label
                    color = (0, 0, 255)

            # draw bounding box and label for the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        else:
            access_state = "denied"
            reason = "no_face"
            label_text = "No face detected"
            color = (255, 255, 0)

        # overlay a small status text
        cv2.putText(frame, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Announce only if access state or reason changes (with cooldown)
        if (access_state != last_state or reason != getattr(last_state, 'reason', None)) and (time() - last_change_time) > STATE_COOLDOWN:
            # Build contextual message
            message = build_context_message(access_state, reason=reason, age_label=age_label)
            speak_async(message)
            last_state = access_state
            last_change_time = time()

        cv2.imshow("Unified Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
