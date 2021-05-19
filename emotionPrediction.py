import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import json
import matplotlib.pyplot as plt
import numpy as np
from humanfriendly import format_timespan
import base64
from io import BytesIO


class Prediction:

    def __init__(self):
        self.loadJsonModel('fer.json')
        self.initCV2()
        self.emotions = ['anger', 'disgust', 'fear',
                         'happiness', 'neutral', 'sadness', 'surprise']
        with open('modelParams.txt') as json_file:
            self.data = json.load(json_file)
        self.report = []
        self.previous = ["", 0, 0.0]

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
        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
        frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while(cap.isOpened()):

            ret, img = cap.read()

            if ret == True:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
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

                    cv2.putText(img, predicted_emotion + " "+str(predictions[0][max_index]), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    resized_img = cv2.resize(img, (1000, 700))
                    #cv2.imshow('Facial Emotion Recognition', resized_img)

                    self.flagCreation(predictions, max_index,
                                      predicted_emotion, frames)

                out.write(img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            if(len(self.report) > 0):
                if(isinstance(self.report[-1][2], int)):
                    self.report[-1][2] = self.timeConversion(
                        (self.report[-1][2]/fps))
            frames += 1
        self.makeReport()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def imagePrediction(self, route, imageCaputured, frames):

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
            cv2.putText(img, predicted_emotion + " "+str(predictions[0][max_index]), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            resized_img = cv2.resize(img, (1000, 700))
            self.flagCreation(predictions, max_index,
                              predicted_emotion, frames)

        """    if route != "":
                cv2.imshow('Facial Emotion Recognition', resized_img)
        if route != "":
            cv2.waitKey(0)
        """
        return img
        # self.cleaningCV2()

    def flagCreation(self, predictions, max_index, predicted_emotion, frames):
        if(predictions[0][max_index] > 0.50):

            print(self.previous)
            if(self.previous[0] == predicted_emotion):
                self.previous[1] += 1
                if(self.previous[2] == 0):
                    self.previous[2] = frames
            else:
                if(self.previous[1] > 3 and (self.previous[0] == "fear" or self.previous[0] == "happiness")):
                    self.report.append(
                        [self.previous[0], self.previous[1], self.previous[2]])

                self.previous[0] = predicted_emotion
                self.previous[1] = 1
                self.previous[2] = 0

    def videoPrediction(self, route):
        cap = cv2.VideoCapture(route)
        out_file = "new"+route
        ret, frame = cap.read()
        video_shape = (int(cap.get(3)), int(cap.get(4)))

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, 20.0, video_shape, True)

        frames = 0
        while ret:
            predict_image = self.imagePrediction("", frame, frames)
            out.write(predict_image)
            ret, frame = cap.read()
            print(self.report)
            if(len(self.report) > 0):
                if(isinstance(self.report[-1][2], int)):
                    self.report[-1][2] = self.timeConversion(
                        (self.report[-1][2]/fps))
            frames += 1

        print(self.report)
        self.makeReport()
        print(out_file + " created")

        self.cleaningCV2()

    def timeConversion(self, time):

        time = format_timespan(time)

        print(time)
        return time

    def cleaningCV2(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def showAccuracy(self):

        for p in self.data['modelParams']:
            #print('accuracy: ' + str(p['accuracy']))
            #print('val_accuracy: ' + str(p['val_accuracy']))
            #print('loss: ' + str(p['loss']))
            #print('val_loss: ' + str(p['val_loss']))
            print('Perdia final del modelo: ' + str(p['lossEvaluate']))
            print('Accuracy final del modelo: ' + str(p['accEvaluate']))

    def trainingGraphics(self):

        fig, ax = plt.subplots(1, 2)
        train_acc = self.data['modelParams'][0]['accuracy']
        train_loss = self.data['modelParams'][0]['loss']
        fig.set_size_inches(12, 4)

        ax[0].plot(self.data['modelParams'][0]['accuracy'])
        ax[0].plot(self.data['modelParams'][0]['val_accuracy'])
        ax[0].set_title('Training Accuracy vs Validation Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Validation'], loc='upper left')

        ax[1].plot(self.data['modelParams'][0]['loss'])
        ax[1].plot(self.data['modelParams'][0]['val_loss'])
        ax[1].set_title('Training Loss vs Validation Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Validation'], loc='upper left')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"

    def makeReport(self):
        data = {}
        data['studentName'] = []
        data['studentName'].append({
            'flags': self.report
        })
        with open('report.txt', 'w') as outfile:
            json.dump(data, outfile)

        self.report = []
        self.previous = ["", 0, 0.0]
