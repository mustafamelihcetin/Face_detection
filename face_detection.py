#face_detection.py
import cv2

print('OpenCV Version: '+cv2.__version__)


# Load the Haar cascade
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture the frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
camera.release()
cv2.destroyAllWindows()
