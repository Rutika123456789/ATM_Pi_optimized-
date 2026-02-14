import tkinter as tk
from tkinter import StringVar
import RPi.GPIO as GPIO
import time

from experiment import run_security_check

# ===============================
# GPIO KEYPAD CONFIG
# ===============================
ROWS = [5, 6, 13, 19]
COLS = [12, 16, 20, 21]

KEYS = [
    ['1','2','3','A'],
    ['4','5','6','B'],
    ['7','8','9','C'],
    ['*','0','#','D']
]

GPIO.setmode(GPIO.BCM)

for r in ROWS:
    GPIO.setup(r, GPIO.OUT)
    GPIO.output(r, GPIO.HIGH)

for c in COLS:
    GPIO.setup(c, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ===============================
# USERS
# ===============================
USERS = {
    'A': '1231',
    'B': '2345',
    'C': '3456',
    'D': '4567'
}

# ===============================
# STATES
# ===============================
STATE_WAIT_SECURITY = 0
STATE_CARD = 1
STATE_PIN = 2
STATE_MENU = 3
STATE_DONE = 4

state = STATE_WAIT_SECURITY
current_user = ""
entered_pin = ""

security_checked = False

# ===============================
# KEYPAD CONTROL VARIABLES
# ===============================
last_key = None
last_press_time = 0
DEBOUNCE_DELAY = 0.25  # seconds

# ===============================
# KEYPAD SCAN (FIXED)
# ===============================
def read_keypad():
    global last_key, last_press_time

    for i, row in enumerate(ROWS):
        GPIO.output(row, GPIO.LOW)

        for j, col in enumerate(COLS):
            if GPIO.input(col) == GPIO.LOW:
                GPIO.output(row, GPIO.HIGH)

                now = time.time()

                if last_key != KEYS[i][j] or (now - last_press_time) > DEBOUNCE_DELAY:
                    last_key = KEYS[i][j]
                    last_press_time = now
                    return KEYS[i][j]

                return None

        GPIO.output(row, GPIO.HIGH)

    last_key = None
    return None

# ===============================
# UI SETUP
# ===============================
root = tk.Tk()
root.title("SBU BANK ATM")
root.geometry("800x480")
root.resizable(False, False)

canvas = tk.Canvas(root, width=800, height=480)
canvas.pack(fill="both", expand=True)

for i in range(480):
    canvas.create_line(0, i, 800, i, fill="#003366")

display_text = StringVar()
display_text.set("Initializing Security...")

label = tk.Label(
    root,
    textvariable=display_text,
    font=("Helvetica", 24, "bold"),
    fg="white",
    bg="#003366",
    justify="center"
)
label.place(relx=0.5, rely=0.4, anchor="center")

# ===============================
# ATM LOOP (NON-BLOCKING)
# ===============================
def atm_loop():
    global state, current_user, entered_pin, security_checked

    key = read_keypad()

    # SECURITY CHECK
    if state == STATE_WAIT_SECURITY:
        if not security_checked:
            display_text.set("Verifying Identity...\nPlease look at camera")
            root.update_idletasks()

            result = run_security_check()
            security_checked = True

            if result == "GRANTED":
                display_text.set("Welcome to SBU Bank\nInsert Card (A-D)")
                state = STATE_CARD
            else:
                display_text.set("Access Denied\nRetrying...")
                security_checked = False

    # CARD INSERT
    elif state == STATE_CARD and key:
        if key in USERS:
            current_user = key
            entered_pin = ""
            display_text.set("Enter PIN")
            state = STATE_PIN

    # PIN ENTRY
    elif state == STATE_PIN and key:
        if key.isdigit():
            entered_pin += key
            display_text.set("Enter PIN\n" + "*" * len(entered_pin))

        elif key == '#':
            if entered_pin == USERS[current_user]:
                display_text.set("1: Balance\n2: Withdraw\n3: Exit")
                state = STATE_MENU
            else:
                display_text.set("Wrong PIN\nInsert Card")
                state = STATE_CARD

    # MENU
    elif state == STATE_MENU and key:
        if key == '1':
            display_text.set("Balance: ₹10,000")
        elif key == '2':
            display_text.set("Withdraw Successful")
        elif key == '3':
            display_text.set("Thank you for banking\nwith SBU Bank")
            state = STATE_DONE

    # EXIT
    elif state == STATE_DONE:
        display_text.set("Insert Card (A-D)")
        state = STATE_CARD

    root.after(80, atm_loop)

# ===============================
# START
# ===============================
atm_loop()
root.mainloop()
GPIO.cleanup()
