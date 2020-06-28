import time
import os
import _pickle as cPickle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from parameters_SVM import TRAININGSVM
from parameter_input import PARAMETERINPUT
from data_loader_all_model_SVM import load_data_svm

def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy

lstModelSVM = [TRAININGSVM.save_model_path_landmarks, TRAININGSVM.save_model_path_landmarks_hog, TRAININGSVM.save_model_path_landmarks_hog_sw]
lstfeature = ["landmarks", "landmarks_and_hog", "landmarks_and_hog_sw"]
for feature in lstfeature:
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        if(feature == "landmarks_and_hog"):
            data, validation, test = load_data_svm(validation=True, test=True, feature=feature)
            if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog):
                with open(TRAININGSVM.save_model_path_landmarks_hog, 'rb') as f:
                    model = cPickle.load(f)
            else:
                print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog))
            print("--")
            print("Validation samples landmarks_and_hog: {}".format(len(validation['Y'])))
            print("Test samples landmarks_and_hog: {}".format(len(test['Y'])))
            print("--")
            print("evaluating...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_faceLandmarksHoG_svm = format(validation_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarksHoG_svm = format(test_accuracy * 100)
            PARAMETERINPUT.Time_faceLandmarksHoG_svm = format(time.time() - start_time)
        elif (feature == "landmarks_and_hog_sw"):
            data, validation, test = load_data_svm(validation=True, test=True, feature=feature)
            if os.path.isfile(TRAININGSVM.save_model_path_landmarks_hog_sw):
                with open(TRAININGSVM.save_model_path_landmarks_hog_sw, 'rb') as f:
                    model = cPickle.load(f)
            else:
                print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks_hog_sw))
            print("--")
            print("Validation samples landmarks_and_hog_sw: {}".format(len(validation['Y'])))
            print("Test samples landmarks_and_hog_sw: {}".format(len(test['Y'])))
            print("--")
            print("evaluating...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_svm= format(validation_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarksHoGSlidingWindow_svm = format(test_accuracy * 100)
            PARAMETERINPUT.Time_faceLandmarksHoGSlidingWindow_svm = format(time.time() - start_time)
        else:
            data, validation, test = load_data_svm(validation=True, test=True, feature=feature)
            if os.path.isfile(TRAININGSVM.save_model_path_landmarks):
                with open(TRAININGSVM.save_model_path_landmarks, 'rb') as f:
                    model = cPickle.load(f)
            else:
                print("Error: file '{}' not found".format(TRAININGSVM.save_model_path_landmarks))
            print("--")
            print("Validation samples landmarks: {}".format(len(validation['Y'])))
            print("Test samples landmarks: {}".format(len(test['Y'])))
            print("--")
            print("evaluating...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_faceLandmarks_svm= format(validation_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarks_svm = format(test_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarks_svm = format(time.time() - start_time)
