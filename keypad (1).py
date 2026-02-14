from gpiozero import DigitalOutputDevice, Button
from time import sleep

# BCM GPIO numbers (change ONLY if your wiring is different)
rows_pins = [5, 6, 13, 19]        # R1 R2 R3 R4
cols_pins = [12, 16, 20, 21]     # C1 C2 C3 C4

# 4x4 keypad layout
keys = [
    "1", "2", "3", "A",
    "4", "5", "6", "B",
    "7", "8", "9", "C",
    "*", "0", "#", "D"
]

# Rows as outputs (inactive LOW), Columns as inputs (active LOW)
rows = [DigitalOutputDevice(pin, initial_value=False) for pin in rows_pins]
cols = [Button(pin, pull_up=False) for pin in cols_pins]

def read_keypad():
    pressed_keys = []
    for i, row in enumerate(rows):
        row.on()  # Activate current row
        for j, col in enumerate(cols):
            if col.is_pressed:
                index = i * len(cols) + j
                pressed_keys.append(keys[index])
        row.off()  # Deactivate row
    return pressed_keys

print("4x4 Keypad test started")
print("Press keys on keypad (CTRL+C to stop)")

last_pressed = []

try:
    while True:
        pressed = read_keypad()
        if pressed and pressed != last_pressed:
            print("Pressed:", pressed)
        last_pressed = pressed
        sleep(0.1)

except KeyboardInterrupt:
    print("\nTest stopped")

finally:
    for row in rows:
        row.off()