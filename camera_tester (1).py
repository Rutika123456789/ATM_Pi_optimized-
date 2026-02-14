import cv2

# Open default camera (USB cam or Pi cam via V4L2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("✅ Camera opened successfully")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
