import cv2

def detect_and_blur_faces():
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video stream (use 0 for the default camera)
    video_stream = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video stream
        ret, frame = video_stream.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face (optional, for visualization)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Blur the detected face region
            face_roi = frame[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)  # You can adjust the blur parameters

            # Place the blurred face back into the original frame
            frame[y:y + h, x:x + w] = blurred_face

        # Display the frame
        cv2.imshow('Face Detection and Blurring', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close all windows
    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_and_blur_faces()
