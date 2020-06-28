from parameters_SVM import DATASET, HYPERPARAMS
from parameter_input import MODELPATHSVM
import numpy as np

def load_data_svm(validation=False, test=False, feature="landmarks_and_hog_sw"):
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":
        # load train set
        if feature == "landmarks_and_hog":
            data_dict['X'] = np.load(MODELPATHSVM.train_folder_landmarks_and_hog + '/landmarks.npy')
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            data_dict['X'] = np.concatenate((data_dict['X'], np.load(MODELPATHSVM.train_folder_landmarks_and_hog + '/hog_features.npy')),
                                            axis=1)
            data_dict['Y'] = np.load(MODELPATHSVM.train_folder_landmarks_and_hog + '/labels.npy')
        elif feature == "landmarks_and_hog_sw":
            data_dict['X'] = np.load(MODELPATHSVM.train_folder_landmarks_and_hog_sw + '/landmarks.npy')
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            data_dict['X'] = np.concatenate((data_dict['X'], np.load(MODELPATHSVM.train_folder_landmarks_and_hog_sw + '/hog_features.npy')),
                                            axis=1)
            data_dict['Y'] = np.load(MODELPATHSVM.train_folder_landmarks_and_hog_sw + '/labels.npy')
        elif feature == "landmarks":
            data_dict['X'] = np.load(MODELPATHSVM.train_folder_landmarks + '/landmarks.npy')
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            data_dict['Y'] = np.load(MODELPATHSVM.train_folder_landmarks + '/labels.npy')
        elif feature == "hog":
            data_dict['X'] = np.load(MODELPATHSVM.train_folder_hog + '/hog_features.npy')
            data_dict['Y'] = np.load(MODELPATHSVM.train_folder_hog + '/labels.npy')
        else:
            print("Error '{}' features not recognized".format(feature))

        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
            # load validation set
            if feature == "landmarks_and_hog":
                validation_dict['X'] = np.load(MODELPATHSVM.validation_folder_landmarks_and_hog + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                validation_dict['X'] = np.concatenate(
                    (validation_dict['X'], np.load(MODELPATHSVM.validation_folder_landmarks_and_hog + '/hog_features.npy')), axis=1)
                validation_dict['Y'] = np.load(MODELPATHSVM.validation_folder_landmarks_and_hog + '/labels.npy')
            elif feature == "landmarks_and_hog_sw":
                validation_dict['X'] = np.load(MODELPATHSVM.validation_folder_landmarks_and_hog_sw + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                validation_dict['X'] = np.concatenate(
                    (validation_dict['X'], np.load(MODELPATHSVM.validation_folder_landmarks_and_hog_sw + '/hog_features.npy')), axis=1)
                validation_dict['Y'] = np.load(MODELPATHSVM.validation_folder_landmarks_and_hog_sw + '/labels.npy')
            elif feature == "landmarks":
                validation_dict['X'] = np.load(MODELPATHSVM.validation_folder_landmarks + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                validation_dict['Y'] = np.load(MODELPATHSVM.validation_folder_landmarks + '/labels.npy')
            elif feature == "hog":
                validation_dict['X'] = np.load(MODELPATHSVM.validation_folder_hog + '/hog_features.npy')
                validation_dict['Y'] = np.load(MODELPATHSVM.validation_folder_hog + '/labels.npy')
            else:
                print("Error '{}' features not recognized".format(feature))

            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load train set
            if feature == "landmarks_and_hog":
                test_dict['X'] = np.load(MODELPATHSVM.test_folder_landmarks_and_hog + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['X'] = np.concatenate((test_dict['X'], np.load(MODELPATHSVM.test_folder_landmarks_and_hog + '/hog_features.npy')),
                                                axis=1)
                test_dict['Y'] = np.load(MODELPATHSVM.test_folder_landmarks_and_hog + '/labels.npy')
                np.save(MODELPATHSVM.test_folder_landmarks_and_hog + "/lab.npy", test_dict['Y'])
            elif feature == "landmarks_and_hog_sw":
                test_dict['X'] = np.load(MODELPATHSVM.test_folder_landmarks_and_hog_sw + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['X'] = np.concatenate(
                    (test_dict['X'], np.load(MODELPATHSVM.test_folder_landmarks_and_hog_sw + '/hog_features.npy')),
                    axis=1)
                test_dict['Y'] = np.load(MODELPATHSVM.test_folder_landmarks_and_hog_sw + '/labels.npy')
                np.save(MODELPATHSVM.test_folder_landmarks_and_hog_sw + "/lab.npy", test_dict['Y'])
            elif feature == "landmarks":
                test_dict['X'] = np.load(MODELPATHSVM.test_folder_landmarks + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['Y'] = np.load(MODELPATHSVM.test_folder_landmarks + '/labels.npy')
                np.save(MODELPATHSVM.test_folder_landmarks + "/lab.npy", test_dict['Y'])
            elif feature == "hog":
                test_dict['X'] = np.load(MODELPATHSVM.test_folder_hog + '/hog_features.npy')
                test_dict['Y'] = np.load(MODELPATHSVM.test_folder_hog + '/labels.npy')
                np.save(MODELPATHSVM.test_folder_hog + "/lab.npy", test_dict['Y'])
            else:
                print("Error '{}' features not recognized".format(feature))
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else:
            return data_dict, validation_dict, test_dict
    else:
        print("Unknown dataset")
        exit()
