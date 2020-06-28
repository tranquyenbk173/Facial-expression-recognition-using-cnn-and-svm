import os

class ParameterInput:
    # SVM
    raw_svm = '--'
    Validation_faceLandmarks_svm = 0
    Validation_faceLandmarksHoG_svm = 0
    Validation_faceLandmarksHoGSlidingWindow_svm = 0

    Test_faceLandmarks_svm = 0
    Test_faceLandmarksHoG_svm = 0
    Test_faceLandmarksHoGSlidingWindow_svm = 0

    Time_faceLandmarks_svm = 0
    Time_faceLandmarksHoG_svm = 0
    Time_faceLandmarksHoGSlidingWindow_svm = 0

    # CNN
    Validation_raw_cnn = 0
    Validation_faceLandmarks_cnn = 0
    Validation_faceLandmarksHoG_cnn = 0
    Validation_faceLandmarksHoGSlidingWindow_cnn = 0

    Test_raw_cnn = 0
    Test_faceLandmarks_cnn = 0
    Test_faceLandmarksHoG_cnn = 0
    Test_faceLandmarksHoGSlidingWindow_cnn = 0

    Time_raw_cnn = 0
    Time_faceLandmarks_cnn = 0
    Time_faceLandmarksHoG_cnn = 0
    Time_faceLandmarksHoGSlidingWindow_cnn = 0

    #SVM
    prediction_svm = "----"
    confidence_svm = 0

    #CNN
    prediction_cnn = "----"
    confidence_cnn = 0

    #path image
    image_path="---"
    feature_name="---"


class ModelPathSVM:
    train_folder_hog = 'fer2013_features/Training'
    validation_folder_hog = 'fer2013_features/PublicTest'
    test_folder_hog = 'fer2013_features/PrivateTest'

    train_folder_landmarks = 'fer2013_features_landmarks/Training'
    validation_folder_landmarks = 'fer2013_features_landmarks/PublicTest'
    test_folder_landmarks = 'fer2013_features_landmarks/PrivateTest'

    train_folder_landmarks_and_hog = 'fer2013_features_landmarks_and_hog/Training'
    validation_folder_landmarks_and_hog = 'fer2013_features_landmarks_and_hog/PublicTest'
    test_folder_landmarks_and_hog = 'fer2013_features_landmarks_and_hog/PrivateTest'

    train_folder_landmarks_and_hog_sw = 'fer2013_features_landmarks_and_hog_sw/Training'
    validation_folder_landmarks_and_hog_sw = 'fer2013_features_landmarks_and_hog_sw/PublicTest'
    test_folder_landmarks_and_hog_sw = 'fer2013_features_landmarks_and_hog_sw/PrivateTest'

class ModelPathCNN:
    train_folder_raw = 'fer2013_features_CNN_raw/Training'
    validation_folder_raw = 'fer2013_features_CNN_raw/PublicTest'
    test_folder_raw = 'fer2013_features_CNN_raw/PrivateTest'

    train_folder_landmarks = 'fer2013_features_CNN_landmarks/Training'
    validation_folder_landmarks = 'fer2013_features_CNN_landmarks/PublicTest'
    test_folder_landmarks = 'fer2013_features_CNN_landmarks/PrivateTest'

    train_folder_landmarks_hog = 'fer2013_features_CNN_landmarks_hog/Training'
    validation_folder_landmarks_hog = 'fer2013_features_CNN_landmarks_hog/PublicTest'
    test_folder_landmarks_hog = 'fer2013_features_CNN_landmarks_hog/PrivateTest'

    train_folder_landmarks_hog_sw = 'fer2013_features_CNN_landmarks_hog_sw/Training'
    validation_folder_landmarks_hog_sw = 'fer2013_features_CNN_landmarks_hog_sw/PublicTest'
    test_folder_landmarks_hog_sw = 'fer2013_features_CNN_landmarks_hog_sw/PrivateTest'

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

PARAMETERINPUT = ParameterInput()
MODELPATHSVM = ModelPathSVM()
MODELPATHCNN = ModelPathCNN()
