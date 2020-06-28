# 20052020
#  - Với 1 ảnh đầu vào gọi là img
# - Ta thực hiện trích xuất landmark của khuôn mặt trong đó bằng dlib và flattern nó ra
# - Ta tiếp tục lấy HOG Feauture của khuôn mặt đó
# - Concat theo axis=1 cái flatterned landmark với cái hog để ra feature vector
# - Chuyển FV đó thành tensor nếu cần và predict


import tensorflow as tf
import time
import numpy as np
import argparse
import dlib
import cv2
import os
import _pickle as cPickle
from sklearn.svm import SVC, LinearSVC
from parameters_SVM import TRAININGSVM, VIDEO_PREDICTOR
from data_loader_SVM import load_data
from sklearn.metrics import accuracy_score

window_size = 24
window_step = 6

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def load_model():
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        data, validation, test = load_data(validation=True, test=True)
        if os.path.isfile(TRAININGSVM.save_model_path):
            with open(TRAININGSVM.save_model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAININGSVM.save_model_path))

        print("--")
        print("Validation samples: {}".format(len(validation['Y'])))
        print("Test samples: {}".format(len(test['Y'])))
        print("--")
        print("evaluating...")
        start_time = time.time()
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        test_accuracy = evaluate(model, test['X'], test['Y'])
        print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
        print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))

    return model


def get_landmarks(image, rects, predictor=predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# Phat hien khuon mat trong anh.
def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, 48, window_step):
        for x in range(0, 48, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False))
    return hog_windows


def predict(image, model, shape_predictor=None):
    # Get hog
    features = sliding_hog_windows(image)

    # Get landmakr
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    face_landmarks = get_landmarks(image, face_rects)

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
    # print(tensor_image.shape)
    # print("---333---")
    # print(*tensor_image, sep=' ')
    # print("---444---")
    # print(*features.reshape((1, -1)), sep=' ')
    # print("---666---")
    # predicted_label = model.predict(tensor_image)
    # print("predicted_label",*predicted_label, sep=' ')

    predicted_label = model.predict_proba(tensor_image)
    return get_emotion(predicted_label[0])


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("test")
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
    label = label.tolist()
    print("lable", label)
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Image file to predict")
args = parser.parse_args()

from skimage.feature import hog

if args.image:
    if os.path.isfile(args.image):
        model = load_model()
        image = cv2.imread(args.image, 0)
        # print("shape_predictor",X)
        start_time = time.time()
        emotion, confidence = predict(image, model, predictor)
        print("emotion")
        total_time = time.time() - start_time
        print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))  # confidence * 100
        print("time: {0:.1f} sec".format(total_time))
    else:
        print("Error: file '{}' not found".format(args.image))
