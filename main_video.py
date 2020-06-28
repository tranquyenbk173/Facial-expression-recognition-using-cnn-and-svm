import sys
import subprocess
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
from tkinter.ttk import *

import cv2
import dlib
from parameters_SVM import  DATASET, VIDEO_PREDICTOR,TRAININGSVM
from predict_SVM import load_model, predict
import os
import _pickle as cPickle
import imutils
import time
import numpy as np
from skimage.feature import hog
from parameter_input import PARAMETERINPUT
from parameters import NETWORK,VIDEO_PREDICTOR,TRAINING, DATASET

import tensorflow as tf
from tflearn import DNN
from model_all_cnn import build_model

window_size = 24
window_step = 6
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#SVM
def get_landmarks_SVM(image, rects, predictor=predictor):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# Phat hien khuon mat trong anh.
def sliding_hog_windows_SVM(image):
    hog_windows = []
    for y in range(0, 48, window_step):
        for x in range(0, 48, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False))
    return hog_windows
def predict_SVM(image, model, shape_predictor=None, use_landmarks=False, use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False):
    if use_hog_sliding_window_and_landmarks:
        # Get hog
        features = sliding_hog_windows_SVM(image)

        # Get landmakr
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks_SVM(image, face_rects)
        # Build vector
        # print(face_landmarks.shape)
        face_landmarks = face_landmarks.flatten()
        X = face_landmarks  # .reshape(136,1)
        # print(X.shape)
        # print(X)
        features = np.array(features).flatten()
        features = features.reshape(1, 2592)
        # print(features.shape)
        X = np.concatenate((X, features), axis=1)
        tensor_image = X  # np.expand_dims(X,axis=2) #X.reshape(-1,) #image.reshape([-1, 48,48, 1])
        predicted_label = model.predict_proba(tensor_image)
        return get_emotion(predicted_label[0])
    elif use_hog_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=48, bottom=48)]
        face_landmarks = np.array([get_landmarks_SVM(image, face_rects, predictor)])
        features = face_landmarks
        hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True)
        hog_features = np.asarray(hog_features)
        face_landmarks = face_landmarks.flatten()
        features = np.concatenate((face_landmarks, hog_features))
        predicted_label = model.predict_proba(features.reshape((1, -1)))
        return get_emotion(predicted_label[0])
    else:
        face_rects = [dlib.rectangle(left=0, top=0, right=48, bottom=48)]
        face_landmarks = np.array([get_landmarks_SVM(image, face_rects, predictor)])
        face_landmarks = face_landmarks.flatten()
        face_landmarks = face_landmarks.reshape(1, 136)
        predicted_label = model.predict_proba(face_landmarks)
        return get_emotion(predicted_label[0])


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
    label = label.tolist()
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)

#CNN
def get_landmarks(image, rects, predictor):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, NETWORK.input_size, window_step):
        for x in range(0, NETWORK.input_size, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualize=False))
    return hog_windows

def predict(image, model, shape_predictor=predictor, use_landmarks=False,
                                               use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False):
    # get landmarks
    if use_landmarks or use_hog_and_landmarks or use_hog_sliding_window_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array([get_landmarks(image, face_rects, shape_predictor)])
        features = face_landmarks
        if use_landmarks and use_hog_and_landmarks==False and use_hog_sliding_window_and_landmarks==False:
            face_landmarks = face_landmarks.flatten()
            features = face_landmarks
            tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            predicted_label = model.predict([tensor_image, features.reshape(1, 68, 2)])
            return get_emotion(predicted_label[0])
        if use_hog_sliding_window_and_landmarks:
            hog_features = sliding_hog_windows(image)
            hog_features = np.asarray(hog_features)
            face_landmarks = face_landmarks.flatten()
            features = np.concatenate((face_landmarks, hog_features))
        else:
            hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True)
            hog_features = np.asarray(hog_features)
            face_landmarks = face_landmarks.flatten()
            features = np.concatenate((face_landmarks, hog_features))
        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict([tensor_image, features.reshape((1, -1))])
        return get_emotion(predicted_label[0])
    else:
        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict(tensor_image)
        return get_emotion(predicted_label[0])
    return None


def checkcmbo():
    if cmbEffects.get() == "FaceLandmarks":
        # SVM
        class EmotionRecognizer:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):

                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)

                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks):
                    with open(TRAININGSVM.save_model_path_landmarks, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict_SVM(image, self.model, self.shape_predictor, use_landmarks=True,
                                                  use_hog_and_landmarks=False,
                                                  use_hog_sliding_window_and_landmarks=False)
                return emotion, confidence

            def recognize_emotions(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_svm = label
                                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_svm = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizer()
        r.recognize_emotions()
        frmTable(root)

    elif cmbEffects.get() == "FaceLandmarksHOG":
        # SVM
        class EmotionRecognizer:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):

                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)

                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog):
                    with open(TRAININGSVM.save_model_path_landmarks_hog, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict_SVM(image, self.model, self.shape_predictor, use_landmarks=False,
                                                  use_hog_and_landmarks=True,
                                                  use_hog_sliding_window_and_landmarks=False)
                return emotion, confidence

            def recognize_emotions(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_svm = label
                                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_svm = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizer()
        r.recognize_emotions()
        frmTable(root)

    elif cmbEffects.get() == "FaceLandmarksHOGSlidingWindow":
        #SVM
        class EmotionRecognizer:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):

                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)

                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog_sw):
                    with open(TRAININGSVM.save_model_path_landmarks_hog_sw, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog_sw))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict_SVM(image, self.model, self.shape_predictor, use_landmarks=True,
                                                  use_hog_and_landmarks=True,
                                                  use_hog_sliding_window_and_landmarks=True)
                return emotion, confidence

            def recognize_emotions(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_svm = label
                                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_svm = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)
                        frmTable(root)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizer()
        r.recognize_emotions()
        frmTable(root)
    elif cmbEffects.get() == "":
        messagebox.showinfo("nothing to show!", "you have to be choose something")

# CNN
def checkcmboCNN():
    if cmbEffectsCNN.get() == "Raw":
        class EmotionRecognizerCNN:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):
                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)
                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)
                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                with tf.Graph().as_default():
                    network = build_model(use_landmarks=False, use_hog_and_landmarks=False,
                                          use_hog_sliding_window_and_landmarks=False)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_raw):
                        model.load(TRAINING.save_model_path_raw)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_raw))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict(image, self.model, self.shape_predictor, use_landmarks=False,
                                              use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False)
                return emotion, confidence

            def recognize_emotionsCNN(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                            else:
                                text = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizerCNN()
        r.recognize_emotionsCNN()
        frmTable(root)
    elif cmbEffectsCNN.get() == "FaceLandmarks":
        class EmotionRecognizerCNN:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):
                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)
                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)
                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                with tf.Graph().as_default():
                    network = build_model(use_landmarks=True, use_hog_and_landmarks=False,
                                          use_hog_sliding_window_and_landmarks=False)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks):
                        model.load(TRAINING.save_model_path_landmarks)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict(image, self.model, self.shape_predictor, use_landmarks=True,
                                              use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False)
                return emotion, confidence

            def recognize_emotionsCNN(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_cnn = label
                                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_cnn = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizerCNN()
        r.recognize_emotionsCNN()
        frmTable(root)

    elif cmbEffectsCNN.get() == "FaceLandmarksHOG":
        class EmotionRecognizerCNN:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):
                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)
                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)
                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                with tf.Graph().as_default():
                    network = build_model(use_landmarks=False, use_hog_and_landmarks=True,
                                          use_hog_sliding_window_and_landmarks=False)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks_hog):
                        model.load(TRAINING.save_model_path_landmarks_hog)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict(image, self.model, self.shape_predictor, use_landmarks=True,
                                              use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=False)
                return emotion, confidence

            def recognize_emotionsCNN(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_cnn = label
                                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_cnn = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizerCNN()
        r.recognize_emotionsCNN()
        frmTable(root)

    elif cmbEffectsCNN.get() == "FaceLandmarksHOGSlidingWindow":
        class EmotionRecognizerCNN:
            BOX_COLOR = (0, 255, 0)
            TEXT_COLOR = (0, 255, 0)

            def __init__(self):
                # initializebevideo stream
                self.video_stream = cv2.VideoCapture(VIDEO_PREDICTOR.camera_source)
                self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)
                self.shape_predictor = None
                self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

                model = None
                with tf.Graph().as_default():
                    network = build_model(use_landmarks=True, use_hog_and_landmarks=True,
                                          use_hog_sliding_window_and_landmarks=True)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks_hog_sw):
                        model.load(TRAINING.save_model_path_landmarks_hog_sw)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog_sw))

                self.model = model
                self.last_predicted_time = 0
                self.last_predicted_confidence = 0
                self.last_predicted_emotion = ""

            def predict_emotion(self, image):
                image.resize([48, 48], refcheck=False)
                emotion, confidence = predict(image, self.model, self.shape_predictor, use_landmarks=True,
                                              use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=True)
                return emotion, confidence

            def recognize_emotionsCNN(self):
                failedFramesCount = 0
                detected_faces = []
                time_last_sent = 0
                while True:
                    grabbed, frame = self.video_stream.read()

                    if grabbed:
                        # detection phase
                        frame = imutils.resize(frame, width=600)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # detect faces
                        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if w < 30 and h < 30:  # skip the small faces (probably false detections)
                                continue

                            # bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                            # try to recognize emotion
                            face = gray[y:y + h, x:x + w].copy()
                            if time.time() - self.last_predicted_time < VIDEO_PREDICTOR.time_to_wait_between_predictions:
                                label = self.last_predicted_emotion
                                confidence = self.last_predicted_confidence
                            else:
                                label, confidence = self.predict_emotion(face)
                                self.last_predicted_emotion = label
                                self.last_predicted_confidence = confidence
                                self.last_predicted_time = time.time()

                            # display and send message by socket
                            if VIDEO_PREDICTOR.show_confidence:
                                text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                                PARAMETERINPUT.prediction_cnn = label
                                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
                            else:
                                text = label
                                PARAMETERINPUT.prediction_cnn = label
                            if label is not None:
                                cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            self.TEXT_COLOR, 2)

                        # display images
                        cv2.imshow("Facial Expression Recognition", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        failedFramesCount += 1
                        if failedFramesCount > 10:
                            print("can't grab frames")
                            break

                self.video_stream.release()
                cv2.destroyAllWindows()

        r = EmotionRecognizerCNN()
        r.recognize_emotionsCNN()
        frmTable(root)

    elif cmbEffectsCNN.get() == "":
        messagebox.showinfo("nothing to show!", "you have to be choose something")

root = Tk()
root.title("Facial expression recognition")
root.geometry("400x200+400+100")
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w, h))
root.resizable(width=True, height=True)

#SVM
cmbEffects = ttk.Combobox(root, width="30",
                          values=("FaceLandmarks", "FaceLandmarksHOG", "FaceLandmarksHOGSlidingWindow"))
cmbEffects.grid(row=1, column=1, pady=5, padx=5, sticky=E)

btnClick = ttk.Button(root, text="SVM", command=checkcmbo)
btnClick.grid(row=1, column=2, pady=5, padx=5, sticky=E)

#CNN

cmbEffectsCNN = ttk.Combobox(root, width="30",
                          values=("Raw", "FaceLandmarks", "FaceLandmarksHOG", "FaceLandmarksHOGSlidingWindow"))
cmbEffectsCNN.grid(row=2, column=1, pady=5, padx=5, sticky=E)

btnClick_CNN = ttk.Button(root, text="CNN", command=checkcmboCNN)
btnClick_CNN.grid(row=2, column=2, pady=5, padx=5, sticky=E)

class frmTable(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.CreateUI()
        self.LoadTable()
        self.grid(sticky=(N, S, W, E), row=5, column=0, columnspan=5)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def CreateUI(self):
        tv = Treeview(self)
        # tv['columns'] = ('raw', 'landmarks','landmarkshog','landmarkshogsw')
        tv['columns'] = ('prediction', 'confidence')
        tv.heading("#0", text='#', anchor='center')
        tv.column("#0", anchor="center", width=150)
        tv.heading('prediction', text='Prediction')
        tv.column('prediction', anchor='center', width=150)
        tv.heading('confidence', text='Confidence')
        tv.column('confidence', anchor='center', width=100)

        tv.grid(sticky=(N, S, W, E))
        self.treeview = tv
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def LoadTable(self):
        if (PARAMETERINPUT.confidence_cnn != 0):
            self.treeview.insert('', 'end', text="CNN",
                                 values=(PARAMETERINPUT.prediction_cnn, PARAMETERINPUT.confidence_cnn))
        if (PARAMETERINPUT.confidence_svm != 0):
            self.treeview.insert('', 'end', text="SVM",
                                 values=(PARAMETERINPUT.prediction_svm, PARAMETERINPUT.confidence_svm))

frmTable(root)

root.mainloop()