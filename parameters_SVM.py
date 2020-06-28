import os

class Dataset:
    name = 'Fer2013'
    # train_folder = 'fer2013_features_landmarks/Training'
    # validation_folder = 'fer2013_features_landmarks/PublicTest'
    # test_folder = 'fer2013_features_landmarks/PrivateTest'
    # train_folder = 'fer2013_features_landmarks_and_hog/Training'
    # validation_folder = 'fer2013_features_landmarks_and_hog/PublicTest'
    # test_folder = 'fer2013_features_landmarks_and_hog/PrivateTest'
    train_folder = 'fer2013_features_landmarks_and_hog_sw/Training'
    validation_folder = 'fer2013_features_landmarks_and_hog_sw/PublicTest'
    test_folder = 'fer2013_features_landmarks_and_hog_sw/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1
    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'


class Hyperparams:
    random_state = 0
    epochs = 10000
    epochs_during_hyperopt = 500
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
    features = "landmarks_and_hog"  # "landmarks" or "hog" or "landmarks_and_hog"
    gamma = 'auto'  # use a float number or 'auto'


class Training:
    save_model = True
    save_model_path = "saved_model_landmarks_and_hog_sw.bin"
    save_model_path_landmarks = "saved_model_landmarks.bin"
    save_model_path_landmarks_hog = "saved_model_landmarks_and_hog.bin"
    save_model_path_landmarks_hog_sw = "saved_model_landmarks_and_hog_sw.bin"



class VideoPredictor:
    # emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = False
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = True
    time_to_wait_between_predictions = 0.5

DATASET = Dataset()
TRAININGSVM = Training()
HYPERPARAMS = Hyperparams()
VIDEO_PREDICTOR = VideoPredictor()
