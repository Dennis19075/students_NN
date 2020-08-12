import cv2
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from plyer import notification

import time

# CATEGORIES = ["sleeping","present", "using_phone","eating", "no_present"]
CATEGORIES = ["present", "no_present"]


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def notif(predic):
    if predic == "no_present":
        notification.notify(
            title='Class status',
            message='Estudiante ausente de la clase.',
            app_icon='./icons/sad.ico',  # e.g. 'C:\\icon_32x32.ico'
            timeout=10,  # seconds
        )

def makeModel(image):
    models = tf.keras.models.load_model("class.model")
    prediction = models.predict([prepare(image)])
    print(int(prediction[0][0]))
    print(CATEGORIES[int(prediction[0][0])])
    predic = CATEGORIES[int(prediction[0][0])]
    notif(predic)


# test1.jpeg
# ----------------------------
video_images = './webcam-frames/'

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#Capture images per 25 frame
frameFrequency=1 #per second

#iterate all frames
total_frame = 0

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    total_frame += 1

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    if total_frame%frameFrequency == 0:
        image_name = video_images +'.jpg'
        
        cv2.imwrite(image_name, frame)
        # print(image_name)
        makeModel(image_name)
    time.sleep(1) #per 5 seconds

    c = cv2.waitKey(1)
    if c == 27 or c==115 or c ==83: #esc for stop
        break