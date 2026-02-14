import tkinter as tk
from tkinter import StringVar
import RPi.GPIO as GPIO
import time
import threading
from multi_test import run_security_check

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
security_result = None

# message timer
message_start_time = 0
MESSAGE_DELAY = 5
message_next_state = STATE_MENU  # default

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
# TIMER LABEL
# ===============================
timer_text = StringVar()
timer_text.set("")

timer_label = tk.Label(
    root,
    textvariable=timer_text,
    font=("Helvetica", 18, "bold"),
    fg="yellow",
    bg="#003366"
)
timer_label.place(relx=1.0, y=0, anchor="ne")

# ===============================
# SECURITY CHECK (ASYNC)
# ===============================
def async_security_check(timeout=20):
    global security_result
    # Call the existing security check (blocking) in a thread
    result = run_security_check(timeout)
    security_result = result
    # Clear the timer immediately when done
    root.after(0, lambda: timer_text.set(""))

# ===============================
# COUNTDOWN FUNCTION
# ===============================
def start_countdown(duration):
    start_time = time.time()

    def update_timer():
        remaining = int(duration - (time.time() - start_time))
        if remaining >= 0 and security_result is None:
            timer_text.set(f"Time left: {remaining}s")
            root.after(200, update_timer)
        else:
            timer_text.set("")  # Ensure timer cleared

    update_timer()

# ===============================
# ATM LOOP
# ===============================
def atm_loop():
    global state, current_user, entered_pin, withdraw_amount
    global security_checked, balance, message_start_time, message_next_state, security_result

    key = read_keypad()
    now = time.time()

    # -------- SECURITY --------
    if state == STATE_WAIT_SECURITY:
        if not security_checked:
            display_text.set("Welcome to SBU Bank\nVerifying Identity...")
            security_result = None  # Reset
            start_countdown(20)  # Start countdown
            threading.Thread(target=async_security_check, args=(20,), daemon=True).start()
            security_checked = True

        # Check if security finished
        if security_result is not None:
            if security_result == "GRANTED":
                display_text.set("Insert Card (A-D)")
                state = STATE_CARD
            else:
                display_text.set("Security Denied")
                message_start_time = now
                message_next_state = STATE_CARD
                state = STATE_MESSAGE

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
                message_next_state = STATE_CARD
                state = STATE_MESSAGE

    # -------- MENU --------
    elif state == STATE_MENU and key:
        if key == '1':
            display_text.set(f"Balance: ₹{balance}")
            message_start_time = now
            message_next_state = STATE_CARD
            state = STATE_MESSAGE
        elif key == '2':
            withdraw_amount = ""
            display_text.set("Enter Amount\n# = Confirm")
            state = STATE_WITHDRAW
        elif key == '3':
            display_text.set("Cash Deposited")
            message_start_time = now
            message_next_state = STATE_CARD
            state = STATE_MESSAGE
        elif key == '4':
            display_text.set("Thank You For Banking With Us!")
            message_start_time = now
            message_next_state = STATE_CARD
            state = STATE_MESSAGE

    # -------- WITHDRAW --------
    elif state == STATE_WITHDRAW and key:
        if key.isdigit():
            withdraw_amount += key
            display_text.set(f"Amount: ₹{withdraw_amount}\n# = Confirm")
        elif key == '#':
            if withdraw_amount:
                amt = int(withdraw_amount)
                if amt % 100 != 0:
                    display_text.set("Amount must be multiple of 100")
                    withdraw_amount = ""
                    message_start_time = now
                    message_next_state = STATE_WITHDRAW
                    state = STATE_MESSAGE
                elif amt > balance:
                    display_text.set("Insufficient Balance")
                    withdraw_amount = ""
                    message_start_time = now
                    message_next_state = STATE_WITHDRAW
                    state = STATE_MESSAGE
                else:
                    balance -= amt
                    display_text.set(f"₹{withdraw_amount} Withdrawn")
                    withdraw_amount = ""
                    message_start_time = now
                    message_next_state = STATE_CARD
                    state = STATE_MESSAGE

    # -------- MESSAGE (AUTO RETURN) --------
    elif state == STATE_MESSAGE:
        if now - message_start_time >= MESSAGE_DELAY:
            if message_next_state == STATE_MENU:
                display_text.set(
                    "1: Balance\n2: Withdraw\n3: Deposit\n4: Exit"
                )
            elif message_next_state == STATE_CARD:
                display_text.set("Insert Card (A-D)")
                current_user = ""
                entered_pin = ""
            elif message_next_state == STATE_WITHDRAW:
                display_text.set("Enter Amount\n# = Confirm")

            state = message_next_state

    root.after(80, atm_loop)

# ===============================
# START
# ===============================
try:
    atm_loop()
    root.mainloop()
finally:
    GPIO.cleanup()
