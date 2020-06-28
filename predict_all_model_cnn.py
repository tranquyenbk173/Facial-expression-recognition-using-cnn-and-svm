import tensorflow as tf
from tflearn import DNN
import time
import os
from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR
from parameter_input import PARAMETERINPUT
from model_all_cnn import build_model
from data_loader_all_model_CNN import load_data_cnn
from data_loader import load_data

def evaluate(model, X, X2, Y, use_landmarks=False):
    if use_landmarks:
        accuracy = model.evaluate([X, X2], Y)
    else:
        accuracy = model.evaluate(X, Y)
    return accuracy[0]

lstfeature = ["landmarks", "landmarks_and_hog", "raw", "landmarks_and_hog_sw"]
for feature in lstfeature:
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        if (feature == "landmarks_and_hog"):
            data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=False, use_hog_and_landmarks=True,use_hog_sliding_window_and_landmarks=False)
            network = build_model(use_landmarks=True, use_hog_and_landmarks=True,use_hog_sliding_window_and_landmarks=False)
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
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'], use_landmarks=True)
            print("  - validation accuracy landmarks_and_hog = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
            print("  - test accuracy landmarks_and_hog = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time landmarks_and_hog = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_faceLandmarksHoG_cnn = "{0:.1f}".format(validation_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarksHoG_cnn = "{0:.1f}".format(test_accuracy * 100)
            PARAMETERINPUT.Time_faceLandmarksHoG_cnn = "{0:.1f}".format(time.time() - start_time)
        elif (feature == "landmarks_and_hog_sw"):
            data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=True,
                                               use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=True)
            network = build_model(use_landmarks=True, use_hog_and_landmarks=True, use_hog_sliding_window_and_landmarks=True)
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
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'], use_landmarks=True)
            print("  - validation accuracy landmarks_and_hog_sw = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
            print("  - test accuracy landmarks_and_hog_sw = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time landmarks_and_hog_sw = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(validation_accuracy * 100)
            PARAMETERINPUT.Test_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(test_accuracy * 100)
            PARAMETERINPUT.Time_faceLandmarksHoGSlidingWindow_cnn = "{0:.1f}".format(time.time() - start_time)
        elif (feature == "landmarks"):
            data, validation, test = load_data_cnn(validation=True, test=True, use_landmarks=True,
                                               use_hog_and_landmarks=False,use_hog_sliding_window_and_landmarks=False)
            network = build_model(use_landmarks=True,use_hog_and_landmarks=False,use_hog_sliding_window_and_landmarks=False )
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
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'], use_landmarks=True)
            print("  - validation accuracy landmarks = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'], use_landmarks=True)
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
            if os.path.isfile(TRAINING.save_model_path):
                model.load(TRAINING.save_model_path)
            else:
                print("Error: file '{}' not found".format(TRAINING.save_model_path))
                exit()


            validation['X2'] = None
            test['X2'] = None

            print("--")
            print("Validation samples: {}".format(len(validation['Y'])))
            print("Test samples: {}".format(len(test['Y'])))
            print("--")
            print("evaluating...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'])
            print("  - validation accuracy raw = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'])
            print("  - test accuracy raw = {0:.1f}".format(test_accuracy * 100))
            print("  - evalution time raw = {0:.1f} sec".format(time.time() - start_time))
            PARAMETERINPUT.Validation_raw_cnn = "{0:.1f}".format(validation_accuracy * 100)
            PARAMETERINPUT.Test_raw_cnn = "{0:.1f}".format(test_accuracy * 100)
            PARAMETERINPUT.Time_raw_cnn = "{0:.1f}".format(time.time() - start_time)