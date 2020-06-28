import sys
import subprocess
import time
import os
import _pickle as cPickle
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
from tkinter.ttk import *
from parameter_input import PARAMETERINPUT
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tflearn import DNN
from parameters_SVM import TRAININGSVM
from parameter_input import PARAMETERINPUT
from data_loader_all_model_SVM import load_data_svm
from model_all_cnn import build_model
from data_loader_all_model_CNN import load_data_cnn
from data_loader import load_data
from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR

root = Tk()
root.title("Facial expression recognition")
root.geometry("400x200+400+100")
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w, h))
root.resizable(width=True, height=True)

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
        tv['columns'] = ('raw', 'landmarks','landmarkshog','landmarkshogsw')
        tv.heading("#0", text='#', anchor='center')
        tv.column("#0", anchor="center", width=150)
        tv.heading('raw', text='Raw')
        tv.column('raw', anchor='center', width=100)
        tv.heading('landmarks', text='Face landmarks')
        tv.column('landmarks', anchor='center', width=100)
        tv.heading('landmarkshog', text='Face landmarks + HOG')
        tv.column('landmarkshog', anchor='center', width=100)
        tv.heading('landmarkshogsw', text='Face landmarks + HOG on slinding window')
        tv.column('landmarkshogsw', anchor='center', width=200)
        tv.grid(sticky=(N, S, W, E))
        self.treeview = tv
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def LoadTable(self):
        # SVM
        def evaluate(model, X, Y):
            predicted_Y = model.predict(X)
            accuracy = accuracy_score(Y, predicted_Y)
            return accuracy

        def evaluate_cnn(model, X, X2, Y, use_landmarks=False):
            if use_landmarks:
                accuracy = model.evaluate([X, X2], Y)
            else:
                accuracy = model.evaluate(X, Y)
            return accuracy[0]

        lstfeature = ["landmarks", "landmarks_and_hog", "landmarks_and_hog_sw"]
        for feature in lstfeature:
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                if (feature == "landmarks_and_hog"):
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
                    PARAMETERINPUT.Validation_faceLandmarksHoG_svm = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarksHoG_svm = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarksHoG_svm = "{0:.1f}".format(time.time() - start_time)
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
                    PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_svm = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarksHoGSlidingWindow_svm = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarksHoGSlidingWindow_svm = "{0:.1f}".format(time.time() - start_time)
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
                    PARAMETERINPUT.Validation_faceLandmarks_svm = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarks_svm = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarks_svm = "{0:.1f}".format(time.time() - start_time)


        #CNN
        lstfeature = ["landmarks", "landmarks_and_hog", "raw", "landmarks_and_hog_sw"]
        for feature in lstfeature:
            model = None
            with tf.Graph().as_default():
                print("loading pretrained model...")
                if (feature == "landmarks_and_hog"):
                    data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=False,
                                                           use_hog_and_landmarks=True,
                                                           use_hog_sliding_window_and_landmarks=False)
                    network = build_model(use_landmarks=True, use_hog_and_landmarks=True,
                                          use_hog_sliding_window_and_landmarks=False)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks_hog):
                        model.load(TRAINING.save_model_path_landmarks_hog)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog))

                    print("--")
                    print("Validation samples: {}".format(len(validation['Y'])))
                    print("Test samples: {}".format(len(test['Y'])))
                    print("--")
                    print("evaluating...")
                    start_time = time.time()
                    validation_accuracy = evaluate_cnn(model, validation['X'], validation['X2'], validation['Y'],
                                                   use_landmarks=True)
                    print("  - validation accuracy landmarks_and_hog = {0:.1f}".format(validation_accuracy * 100))
                    test_accuracy = evaluate_cnn(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
                    print("  - test accuracy landmarks_and_hog = {0:.1f}".format(test_accuracy * 100))
                    print("  - evalution time landmarks_and_hog = {0:.1f} sec".format(time.time() - start_time))
                    PARAMETERINPUT.Validation_faceLandmarksHoG_cnn = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarksHoG_cnn = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarksHoG_cnn = "{0:.1f}".format(time.time() - start_time)
                elif (feature == "landmarks_and_hog_sw"):
                    data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=True,
                                                           use_hog_and_landmarks=True,
                                                           use_hog_sliding_window_and_landmarks=True)
                    network = build_model(use_landmarks=True, use_hog_and_landmarks=True,
                                          use_hog_sliding_window_and_landmarks=True)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks_hog_sw):
                        model.load(TRAINING.save_model_path_landmarks_hog_sw)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks_hog_sw))

                    print("--")
                    print("Validation samples: {}".format(len(validation['Y'])))
                    print("Test samples: {}".format(len(test['Y'])))
                    print("--")
                    print("evaluating...")
                    start_time = time.time()
                    validation_accuracy = evaluate_cnn(model, validation['X'], validation['X2'], validation['Y'],
                                                   use_landmarks=True)
                    print("  - validation accuracy landmarks_and_hog_sw = {0:.1f}".format(validation_accuracy * 100))
                    test_accuracy = evaluate_cnn(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
                    print("  - test accuracy landmarks_and_hog_sw = {0:.1f}".format(test_accuracy * 100))
                    print("  - evalution time landmarks_and_hog_sw = {0:.1f} sec".format(time.time() - start_time))
                    PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(
                        validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(time.time() - start_time)
                elif (feature == "landmarks"):
                    data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=True,
                                                           use_hog_and_landmarks=False,
                                                           use_hog_sliding_window_and_landmarks=False)
                    network = build_model(use_landmarks=True, use_hog_and_landmarks=False,
                                          use_hog_sliding_window_and_landmarks=False)
                    model = DNN(network)
                    if os.path.isfile(TRAINING.save_model_path_landmarks):
                        model.load(TRAINING.save_model_path_landmarks)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_landmarks))

                    print("--")
                    print("Validation samples: {}".format(len(validation['Y'])))
                    print("Test samples: {}".format(len(test['Y'])))
                    print("--")
                    print("evaluating...")
                    start_time = time.time()
                    validation_accuracy = evaluate_cnn(model, validation['X'], validation['X2'], validation['Y'],
                                                   use_landmarks=True)
                    print("  - validation accuracy landmarks = {0:.1f}".format(validation_accuracy * 100))
                    test_accuracy = evaluate_cnn(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
                    print("  - test accuracy landmarks = {0:.1f}".format(test_accuracy * 100))
                    print("  - evalution time landmarks = {0:.1f} sec".format(time.time() - start_time))
                    PARAMETERINPUT.Validation_faceLandmarks_cnn = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_faceLandmarks_cnn = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_faceLandmarks_cnn = "{0:.1f}".format(time.time() - start_time)

                else:
                    data, validation, test = load_data(validation=True, test=True)
                    network = build_model()
                    model = DNN(network)
                    # Testing phase : load saved model and evaluate on test dataset
                    print("start evaluation...")
                    print("loading pretrained model...")
                    if os.path.isfile(TRAINING.save_model_path_raw):
                        model.load(TRAINING.save_model_path_raw)
                    else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path_raw))
                        exit()

                    if not NETWORK.use_landmarks:
                        validation['X2'] = None
                        test['X2'] = None

                    print("--")
                    print("Validation samples: {}".format(len(validation['Y'])))
                    print("Test samples: {}".format(len(test['Y'])))
                    print("--")
                    print("evaluating...")
                    start_time = time.time()
                    validation_accuracy = evaluate_cnn(model, validation['X'], validation['X2'], validation['Y'])
                    print("  - validation accuracy raw = {0:.1f}".format(validation_accuracy * 100))
                    test_accuracy = evaluate_cnn(model, test['X'], test['X2'], test['Y'])
                    print("  - test accuracy raw = {0:.1f}".format(test_accuracy * 100))
                    print("  - evalution time raw = {0:.1f} sec".format(time.time() - start_time))
                    PARAMETERINPUT.Validation_raw_cnn = "{0:.1f}".format(validation_accuracy * 100)
                    PARAMETERINPUT.Test_raw_cnn = "{0:.1f}".format(test_accuracy * 100)
                    PARAMETERINPUT.Time_raw_cnn = "{0:.1f}".format(time.time() - start_time)

        self.treeview.insert('', 'end', text="CNN", values=(PARAMETERINPUT.Validation_raw_cnn,PARAMETERINPUT.Validation_faceLandmarks_cnn,PARAMETERINPUT.Validation_faceLandmarksHoG_cnn,PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_cnn))
        self.treeview.insert('', 'end', text="SVM", values=('----',PARAMETERINPUT.Validation_faceLandmarks_svm,PARAMETERINPUT.Validation_faceLandmarksHoG_svm,PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_svm))

    # closeButton = ttk.Button(root, text="Close", command=exit).grid(row=6, column=2, pady=5, padx=5, sticky=E)


frmTable(root)

root.mainloop()