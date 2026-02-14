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
# USERS & BALANCE
# ===============================
USERS = {'A':'1234','B':'2345','C':'3456','D':'4567'}
balance = 10000

# ===============================
# STATES
# ===============================
STATE_WAIT_SECURITY = 0
STATE_CARD = 1
STATE_PIN = 2
STATE_MENU = 3
STATE_WITHDRAW = 4
STATE_MESSAGE = 5

state = STATE_WAIT_SECURITY
current_user = ""
entered_pin = ""
withdraw_amount = ""
security_checked = False

# message timer
message_start_time = 0
MESSAGE_DELAY = 5

# ===============================
# KEYPAD CONTROL
# ===============================
last_key = None
last_press_time = 0
DEBOUNCE = 0.25

def read_keypad():
    global last_key, last_press_time

    for r in ROWS:
        GPIO.output(r, GPIO.HIGH)

    for i, row in enumerate(ROWS):
        GPIO.output(row, GPIO.LOW)
        time.sleep(0.002)

        for j, col in enumerate(COLS):
            if GPIO.input(col) == GPIO.LOW:
                key = KEYS[i][j]
                now = time.time()
                if key != last_key and (now - last_press_time) > DEBOUNCE:
                    last_key = key
                    last_press_time = now
                    GPIO.output(row, GPIO.HIGH)
                    return key
        GPIO.output(row, GPIO.HIGH)

    last_key = None
    return None

# ===============================
# UI
# ===============================
root = tk.Tk()
root.geometry("800x480")
root.title("SBU BANK ATM")

display_text = StringVar()
display_text.set("Initializing Security...")

label = tk.Label(
    root,
    textvariable=display_text,
    font=("Helvetica",24,"bold"),
    fg="white",
    bg="#003366",
    justify="center"
)
label.pack(expand=True, fill="both")

# ===============================
# ATM LOOP
# ===============================
def atm_loop():
    global state, current_user, entered_pin, withdraw_amount
    global security_checked, balance, message_start_time

    key = read_keypad()
    now = time.time()

    # -------- SECURITY --------
    if state == STATE_WAIT_SECURITY:
        if not security_checked:
            display_text.set("Verifying Identity...")
            root.update_idletasks()
            if run_security_check() == "GRANTED":
                display_text.set("Insert Card (A-D)")
                state = STATE_CARD
            security_checked = True

    # -------- CARD --------
    elif state == STATE_CARD and key:
        if key in USERS:
            current_user = key
            entered_pin = ""
            display_text.set("Enter PIN")
            state = STATE_PIN

    # -------- PIN --------
    elif state == STATE_PIN and key:
        if key.isdigit():
            entered_pin += key
            display_text.set("Enter PIN\n" + "*" * len(entered_pin))

        elif key == '#':
            if entered_pin == USERS[current_user]:
                display_text.set(
                    "1: Balance\n2: Withdraw\n3: Deposit\n4: Exit"
                )
                state = STATE_MENU
            else:
                display_text.set("Wrong PIN")
                message_start_time = now
                state = STATE_MESSAGE

    # -------- MENU --------
    elif state == STATE_MENU and key:
        if key == '1':
            display_text.set(f"Balance: ₹{balance}")
            message_start_time = now
            state = STATE_MESSAGE

        elif key == '2':
            withdraw_amount = ""
            display_text.set("Enter Amount\n# = Confirm")
            state = STATE_WITHDRAW

        elif key == '3':
            display_text.set("Cash Deposited")
            message_start_time = now
            state = STATE_MESSAGE

        elif key == '4':
            display_text.set("Thank You")
            message_start_time = now
            state = STATE_MESSAGE

    # -------- WITHDRAW --------
    elif state == STATE_WITHDRAW and key:
        if key.isdigit():
            withdraw_amount += key
            display_text.set(f"Amount: ₹{withdraw_amount}\n# = Confirm")
        elif key == '#':
            if withdraw_amount:
                balance -= int(withdraw_amount)
                display_text.set(f"₹{withdraw_amount} Withdrawn")
                message_start_time = now
                state = STATE_MESSAGE

    # -------- MESSAGE (AUTO RETURN) --------
    elif state == STATE_MESSAGE:
        if now - message_start_time >= MESSAGE_DELAY:
            display_text.set(
                "1: Balance\n2: Withdraw\n3: Deposit\n4: Exit"
            )
            state = STATE_MENU

    root.after(80, atm_loop)

# ===============================
# START
# ===============================
try:
    atm_loop()
    root.mainloop()
finally:
    GPIO.cleanup()
