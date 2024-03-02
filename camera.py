import numpy as np
import cv2
import math
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import datetime
from threading import Thread
# from Spotipy import *
import time
import pandas as pd
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6


cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry",     1: "Happy", 2: "Neutral",
                3: "Sad"}
music_dict = {0: 'songs/angry.csv', 1: 'songs/happy__labelled_uri.csv', 2: 'songs/neutral__labelled_uri.csv',
              3: 'songs/sad__labelled_uri.csv'}
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''


class WebcamVideoStream:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


''' Class for reading video stream, generating prediction and recommendations '''


class VideoCamera(object):

    def get_frame(self):
        global cap1
        global df1
        json_file = open('emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        cap1 = WebcamVideoStream(src=0)
        frame = cap1.read()
        image = cv2.resize(frame, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        df1 = pd.read_csv(music_dict[show_text[0]])
        df1 = df1[['uri']]
        random_number = np.random.randint(10, 20)
        df1 = df1.head(random_number)
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
