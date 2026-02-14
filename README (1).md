## 1. System Requirements

**Hardware:**

* Raspberry Pi 4 Model B (2 GB / 4 GB / 8 GB)
* Official Raspberry Pi Camera Module or USB Webcam
* Relay Module + Solenoid Lock
* Bluetooth Speaker (for audio output)
* Internet connection (optional for gTTS)

**Software:**

* Raspberry Pi OS (Bookworm or Bullseye)
* Python 3.11 (used via virtual environment)
* TensorFlow Lite-optimized models

---

##  2. Installing Python 3.11



```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

Verify installation:

```bash
python3.11 --version
```

---

##  3. Create and Activate Virtual Environment

```bash
cd ~/ATM_Security_System
python3.11 -m venv venv
source venv/bin/activate
```

The `(venv)` prefix in your terminal confirms the environment is active.

---

## 📦 4. Install Dependencies

Make sure `requirements.txt` is in the same folder, then run:

```bash
pip install --upgrade pip
pip install -r raspirequirements.txt
```

If any system packages are missing, install them:

```bash
sudo apt install -y espeak libespeak-ng1 portaudio19-dev
```

---


### Install eSpeak engine:

```bash
sudo apt install espeak -y
```






## . Running the Project

Activate your virtual environment:

```bash
cd ~/ATM_Security_System
source venv/bin/activate
```

Then run:

```bash
# Test age detection module
python3.11 AgeClass_WebcamOnly.py

# Test mask detection module
python3.11 detect_and_alert.py

# Run the integrated full system
python3.11 integrated.py
```

---

## 

---

Would you like me to include a **diagram (block or connection layout)** in this README to make it visually complete for presentation or documentation purposes?
