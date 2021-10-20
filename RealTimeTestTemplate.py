# Imports
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model(r'Model Path')
print(model.summary())

haarCascadePath = r'Haar Cascade Path'
targetSize = (224, 224)

webcam = cv2.VideoCapture(0)
labels = pd.read_excel('labels.xlsx')['label']

faceCascade = cv2.CascadeClassifier(f'{haarCascadePath}\\haarcascade_face.xml')
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    _, frame = webcam.read() # Read Image From WebCam
    frame = cv2.flip(frame, 1)

    # Detect Faces
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=5)

    # x, y: Left-Top Coordinate Of Face w,h: Width-Height Of Face
    for (x, y, w, h) in faces:
        # Preprocessing
        image = cv2.resize(frame[x:x+w, y:y+h], targetSize, cv2.INTER_AREA)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Prediction
        pred = model.predict_generator(image, verbose=1)
        print(pred)

        # Binary?
        pred[pred > .5] = 1
        pred[pred <= .5] = 0
        predLabel = labels[int(pred[0][0])]

        # Draw A Rectangle On Face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

        # Show On Camera
        cv2.putText(frame, str(predLabel), (250, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 50, 255))

    cv2.imshow('Recognition...', frame)