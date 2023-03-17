import cv2
import numpy as np
import tkinter as tk

class VideoCaptureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Capture App")
        
        # Create a button widget
        self.btn_snapshot = tk.Button(self.window, text="Capture", command=self.capture_snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        # Initialize the video capture object
        self.cap = cv2.VideoCapture(0)
        
    def capture_snapshot(self):
        # Read a frame from the video feed
        ret, frame = self.cap.read()
        
        if ret:
            # Save the frame as an image
            cv2.imwrite('snapshot.jpg', frame)
            print("Snapshot captured.")
    
    def __del__(self):
        # Release the video capture object when the window is closed
        if self.cap:
            self.cap.release()

# Create the main window
window = tk.Tk()

# Create the video capture app
app = VideoCaptureApp(window)

cap = cv2.VideoCapture(0)

# Run the main event loop
window.mainloop()

if not cap.isOpened():
    print("Unable to open camera")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

    # Display the captured image
    cv2.imshow('Camera', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()