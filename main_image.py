import sys
import subprocess
import os
import numpy as np
import tensorflow as tf
from tflearn import DNN
import argparse
import dlib
import cv2
import time
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
from tkinter.ttk import *
from parameter_input import PARAMETERINPUT
from parameters import NETWORK,VIDEO_PREDICTOR,TRAINING, DATASET
from skimage.feature import hog
from model_all_cnn import build_model
from data_loader_all_model_SVM import load_data_svm
from parameters_SVM import TRAININGSVM
import _pickle as cPickle
from sklearn.metrics import accuracy_score

window_size = 24
window_step = 6

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def openfilename():
    filename = filedialog.askopenfilename(title='"pen')
    return filename


def open_img():
    x = openfilename()
    # print("path",x);
    PARAMETERINPUT.image_path = x
    img = Image.open(x)
    # img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=3, column=1, padx=5, pady=5)


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

def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print( "- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
                label[0]*100, label[1]*100, label[2]*100, label[3]*100, label[4]*100))
    label = label.tolist()
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)

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
        print(face_landmarks.shape)
        face_landmarks = face_landmarks.flatten()
        X = face_landmarks  # .reshape(136,1)
        print(X.shape)
        # print(X)
        features = np.array(features).flatten()
        features = features.reshape(1, 2592)
        # print(features.shape)
        X = np.concatenate((X, features), axis=1)
        tensor_image = X  # np.expand_dims(X,axis=2) #X.reshape(-1,) #image.reshape([-1, 48,48, 1])
        predicted_label = model.predict_proba(tensor_image)
        return get_emotion(predicted_label[0])
    elif use_hog_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
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
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array([get_landmarks_SVM(image, face_rects, predictor)])
        face_landmarks = face_landmarks.flatten()
        face_landmarks = face_landmarks.reshape(1, 136)
        predicted_label = model.predict_proba(face_landmarks)
        return get_emotion(predicted_label[0])


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


def checkcmbo():
    PARAMETERINPUT.feature_name = cmbEffects.get()
    if cmbEffects.get() == "Raw":
        # CNN
        def load_model():
            model = None
            with tf.Graph().as_default():
                network = build_model(use_landmarks=False, use_hog_and_landmarks=False,
                                      use_hog_sliding_window_and_landmarks=False)
                model = DNN(network)
                if os.path.isfile(TRAINING.save_model_path_raw):
                    model.load(TRAINING.save_model_path_raw)
                else:
                    print("Error: file '{}' not found".format(TRAINING.save_model_path_raw))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # shape_predictor = dlib.shape_predictor(predictor)
                start_time = time.time()
                emotion, confidence = predict(image, model, shape_predictor=predictor, use_landmarks=False,
                                              use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False)
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_cnn = emotion
                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))

            #SVM
            PARAMETERINPUT.prediction_svm = "----"
            PARAMETERINPUT.confidence_svm = "----"
        frmTable(root)
    elif cmbEffects.get() == "FaceLandmarks":
        # CNN
        def load_model():
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                network = build_model(use_landmarks=True, use_hog_and_landmarks=False,
                                      use_hog_sliding_window_and_landmarks=False)
                model = DNN(network)
                if os.path.isfile(TRAINING.save_model_path_landmarks):
                    model.load(TRAINING.save_model_path_landmarks)
                else:
                    print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
                start_time = time.time()
                emotion, confidence = predict(image, model, shape_predictor=predictor, use_landmarks=True,
                                              use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False)
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_cnn = emotion
                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))

        #SVM
        def load_model():
            model = None
            with tf.Graph().as_default():
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks):
                    with open(TRAININGSVM.save_model_path_landmarks, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # print("shape_predictor",X)
                start_time = time.time()
                emotion, confidence = predict_SVM(image, model, DATASET.shape_predictor_path,use_landmarks=True, use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=False)
                print("emotion")
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))  # confidence * 100
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_svm = "{0}".format(emotion)
                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))
        frmTable(root)

    elif cmbEffects.get() == "FaceLandmarksHOG":
        #CNN
        def load_model():
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                network = build_model(use_landmarks=True, use_hog_and_landmarks=True,
                                      use_hog_sliding_window_and_landmarks=False)
                model = DNN(network)
                if os.path.isfile(TRAINING.save_model_path_landmarks_hog):
                    model.load(TRAINING.save_model_path_landmarks_hog)
                else:
                    print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # shape_predictor = dlib.shape_predictor(predictor)
                start_time = time.time()
                emotion, confidence = predict(image, model, shape_predictor=predictor , use_landmarks=True,
                                              use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=False)
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_cnn = emotion
                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))

        # SVM
        def load_model():
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog):
                    with open(TRAININGSVM.save_model_path_landmarks_hog, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # print("shape_predictor",X)
                start_time = time.time()
                emotion, confidence = predict_SVM(image, model, DATASET.shape_predictor_path,use_landmarks=False, use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=False)
                print("emotion")
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))  # confidence * 100
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_svm = "{0}".format(emotion)
                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))
        frmTable(root)
    elif cmbEffects.get() == "FaceLandmarksHOGSlidingWindow":
        #CNN
        def load_model():
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                network = build_model(use_landmarks=True, use_hog_and_landmarks=True,
                                      use_hog_sliding_window_and_landmarks=True)
                model = DNN(network)
                if os.path.isfile(TRAINING.save_model_path_landmarks_hog_sw):
                    model.load(TRAINING.save_model_path_landmarks_hog_sw)
                else:
                    print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog_sw))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
                start_time = time.time()
                emotion, confidence = predict(image, model, shape_predictor, use_landmarks=True,
                                              use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=True)
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_cnn = "{0}".format(emotion)
                PARAMETERINPUT.confidence_cnn = "{0:.1f}%".format(confidence * 100)
                # frmTable(root)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))

        #SVM
        def load_model():
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog_sw):
                    with open(TRAININGSVM.save_model_path_landmarks_hog_sw, 'rb') as f:
                        model = cPickle.load(f)
                else:
                    print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog_sw))
            return model

        if PARAMETERINPUT.image_path:
            if os.path.isfile(PARAMETERINPUT.image_path):
                model = load_model()
                image = cv2.imread(PARAMETERINPUT.image_path, 0)
                # print("shape_predictor",X)
                start_time = time.time()
                emotion, confidence = predict_SVM(image, model, DATASET.shape_predictor_path,use_landmarks=False, use_hog_and_landmarks=False, use_hog_sliding_window_and_landmarks=True)
                print("emotion")
                total_time = time.time() - start_time
                print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))  # confidence * 100
                print("time: {0:.1f} sec".format(total_time))
                PARAMETERINPUT.prediction_svm = "{0}".format(emotion)
                PARAMETERINPUT.confidence_svm = "{0:.1f}%".format(confidence * 100)
            else:
                print("Error: file '{}' not found".format(PARAMETERINPUT.image_path))
        frmTable(root)
    elif cmbEffects.get() == "":
        messagebox.showinfo("nothing to show!", "you have to be choose something")

root = Tk()
root.title("Facial expression recognition")
root.geometry("400x200+400+100")
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w, h))
root.resizable(width=True, height=True)

cmbEffects = ttk.Combobox(root, width="30",
                          values=("Raw", "FaceLandmarks", "FaceLandmarksHOG", "FaceLandmarksHOGSlidingWindow"))
cmbEffects.grid(row=1, column=1, pady=5, padx=5, sticky=E)

btnClick = ttk.Button(root, text="Click", command=checkcmbo)
btnClick.grid(row=1, column=2, pady=5, padx=5, sticky=E)

btnImage = ttk.Button(root, text='Open image', command=open_img).grid(row=1, column=0, pady=5, padx=5)


# labelCNN = ttk.Label(root, text="CNN")
# labelCNN.grid(column=0, row=5, ipadx=5, pady=5, sticky=W + N)
#
# labelSVM = ttk.Label(root, text="SVM")
# labelSVM.grid(column=0, row=6, ipadx=5, pady=5, sticky=W + S)

# entryCNN = ttk.Entry(root, width=20)
# entrySVM = ttk.Entry(root, width=20)
#
# entryCNN.grid(column=1, row=5, padx=5, pady=5, sticky=N)
# entrySVM.grid(column=1, row=6, padx=5, pady=5, sticky=S)


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
        if(PARAMETERINPUT.confidence_cnn !=0):
            if(PARAMETERINPUT.feature_name =="Raw"):
                self.treeview.insert('', 'end', text="CNN", values=(PARAMETERINPUT.prediction_cnn, PARAMETERINPUT.confidence_cnn))
                self.treeview.insert('', 'end', text="SVM", values=("----",  "----"))
            else:
                self.treeview.insert('', 'end', text="CNN",
                                     values=(PARAMETERINPUT.prediction_cnn, PARAMETERINPUT.confidence_cnn))
                self.treeview.insert('', 'end', text="SVM",
                                     values=(PARAMETERINPUT.prediction_svm, PARAMETERINPUT.confidence_svm))
    # closeButton = ttk.Button(root, text="Close", command=exit).grid(row=6, column=2, pady=5, padx=5, sticky=E)


frmTable(root)

root.mainloop()