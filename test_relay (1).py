import cv2
import RPi.GPIO as GPIO
import time
import sys

RELAY_PIN = 17  

GPIO.setmode(GPIO.BCM)  # BCM pin numbering
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Ensure relay is off initially


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible. Check connection or try another index (1, 2...).")
    GPIO.cleanup()
    sys.exit(1)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("\nSystem initialized successfully.")
print("Press 'q' to quit safely.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame not captured. Retrying...")
            time.sleep(0.1)
            continue

        # Resize for performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Detect people in the frame
        boxes, weights = hog.detectMultiScale(frame_resized,
                                              winStride=(8, 8),
                                              padding=(16, 16),
                                              scale=1.05)

        person_count = len(boxes)

        # Draw bounding boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display count
        cv2.putText(frame_resized, f"People detected: {person_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        
        if person_count > 1:
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # Activate relay
            cv2.putText(frame_resized, "RELAY ON", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)  # Deactivate relay
            cv2.putText(frame_resized, "RELAY OFF", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        
        cv2.imshow("ATM Security Feed", frame_resized)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Cleaning up...")

finally:
   
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Ensure relay is turned off
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. GPIO cleaned up. Goodbye.")
