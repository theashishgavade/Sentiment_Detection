import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import smtplib
import os
from collections import Counter
from datetime import datetime

# Load the face classifier and emotion classifier
face_classifier = cv2.CascadeClassifier('/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/haarcascade_frontalface_default.xml')
classifier = load_model('/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/Emotion_little_vgg.h5')

# Define class labels for emotions
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_count = Counter()

def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return (0, 0, 0, 0), np.zeros((50, 50), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception:
            return (x, w, y, h), np.zeros((50, 50), np.uint8), img

    return (x, w, y, h), roi_gray, img

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the ROI and lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            emotion_count[label] += 1  # Update emotion count
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the most common emotion
    if emotion_count:
        most_common_emotion = emotion_count.most_common(1)[0][0]
        cv2.putText(frame, f'Most Common: {most_common_emotion}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to take a screenshot
        screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(screenshot_filename, frame)
        print(f"Screenshot saved as {screenshot_filename}")

# Email notification logic
sender_mail = 'sender@fromdomain.com'
receivers_mail = ['receiver@todomain.com']
message = """From: From Person <%s>  
To: To Person <%s>  
Subject: Emotion Detection Notification   
Most Common Emotion Detected: %s  
""" % (sender_mail, ', '.join(receivers_mail), most_common_emotion)

try:
    smtpObj = smtplib.SMTP('localhost')
    smtpObj.sendmail(sender_mail, receivers_mail, message)
    print("Successfully sent email")
except Exception as e:
    print("Error: unable to send email:", str(e))

# Release resources
cap.release()
cv2.destroyAllWindows()
