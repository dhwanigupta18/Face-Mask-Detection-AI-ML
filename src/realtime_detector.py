"""
realtime_detector.py
Run real-time face mask detection using a trained Keras model and OpenCV's Haar cascade for face detection.
Usage:
    python src/realtime_detector.py --model models/mask_detector.h5
"""
import os
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def main(args):
    model_path = args.model
    if not os.path.exists(model_path):
        raise ValueError("Model file not found: %s" % model_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            face = orig[y:y+h, x:x+w]
            face = cv2.resize(face, (224,224))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)[0][0]
            label = "Mask" if pred < 0.5 else "No Mask"
            color = (0,255,0) if label=="Mask" else (0,0,255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

        cv2.imshow("Face Mask Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained Keras model (.h5)")
    args = parser.parse_args()
    main(args)
