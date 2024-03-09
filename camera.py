import numpy as np
import cv2
import math
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
# from Spotipy import *
import pandas as pd
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry",     1: "Happy", 2: "Neutral",
                3: "Sad"}
music_dict = {0: 'songs/angry.csv', 1: 'songs/happy__labelled_uri.csv', 2: 'songs/neutral__labelled_uri.csv',
              3: 'songs/sad__labelled_uri.csv'}
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


class VideoCamera(object):

    def get_frame(self):
        global cap1
        global df1
        json_file = open('emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        cap1 = cv2.VideoCapture(1)
        success, frame = cap1.read()
        image = cv2.resize(frame, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights("emotion_model.h5")
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex
            # print("===========================================",music_dist[show_text[0]],"===========================================")
            # print(df1)
            cv2.putText(image, emotion_dict[maxindex], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            df1 = music_rec()

        global last_frame1
        last_frame1 = image.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1


def music_rec():
    # print('---------------- Value ------------', music_dist[show_text[0]])
    df = pd.read_csv(music_dict[show_text[0]])
    df = df[['uri']]
    random_number = np.random.randint(10, 20)
    df = df.head(random_number)
    return df
