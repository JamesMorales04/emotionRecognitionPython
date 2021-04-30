

import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np


class Prediction:

    def __init__(self):
        self.loadJsonModel('fer.json')
        self.initCV2()
        self.emotions = ['anger', 'disgust', 'fear',
                         'happiness', 'neutral', 'sadness', 'surprise']

    def loadJsonModel(self, route):

        json_file = open(route, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('fer.h5')

    def initCV2(self):
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)

    def liveCamPredict(self):

        self.cap = cv2.VideoCapture(0)
        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_haar_cascade.detectMultiScale.detectMultiScale(gray_img, scaleFactor=1.05,
                                                                                      minNeighbors=5, minSize=(30, 30),
                                                                                      flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(img, (x, y), (x + w, y + h),
                              (0, 255, 0), thickness=2)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0

                predictions = self.model.predict(img_pixels)
                max_index = int(np.argmax(predictions))

                predicted_emotion = self.emotions[max_index]

                print(predicted_emotion)
                cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                resized_img = cv2.resize(img, (1000, 700))
                cv2.imshow('Facial Emotion Recognition', resized_img)

        self.cleaningCV2()

    def imagePrediction(self, route, imageCaputured):

        img = None
        if route == "":
            img = imageCaputured
        else:
            img = cv2.imread(route)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                                                 minNeighbors=5, minSize=(30, 30),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = self.model.predict(img_pixels)
            max_index = int(np.argmax(predictions))

            predicted_emotion = self.emotions[max_index]
            print(predicted_emotion)
            cv2.putText(img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return img
        self.cleaningCV2()

    def videoPrediction(self, route):
        cap = cv2.VideoCapture(route)
        out_file = "new"+route
        ret, frame = cap.read()
        video_shape = (int(cap.get(3)), int(cap.get(4)))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, 20.0, video_shape, True)

        while ret:
            predict_image = self.imagePrediction("", frame)
            out.write(predict_image)
            ret, frame = cap.read()

        print(out_file + " created")

        self.cleaningCV2()

    def cleaningCV2(self):
        self.cap.release()
        cv2.destroyAllWindows()
