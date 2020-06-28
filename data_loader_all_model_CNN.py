from parameters import DATASET, NETWORK
from parameter_input import PARAMETERINPUT, MODELPATHCNN
import numpy as np


def load_data_cnn(validation=False, test=False, use_landmarks=False, use_hog_and_landmarks=False,use_hog_sliding_window_and_landmarks=False):
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":

        # load train set
        # data_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
        # data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        if use_landmarks:
            data_dict['X'] = np.load(MODELPATHCNN.train_folder_landmarks + '/images.npy')
            data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            data_dict['X2'] = np.load(MODELPATHCNN.train_folder_landmarks + '/landmarks.npy')
            data_dict['Y'] = np.load(MODELPATHCNN.train_folder_landmarks + '/labels.npy')
        if use_hog_and_landmarks:
            data_dict['X'] = np.load(MODELPATHCNN.train_folder_landmarks_hog + '/images.npy')
            data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            data_dict['X2'] = np.load(MODELPATHCNN.train_folder_landmarks_hog + '/landmarks.npy')
            data_dict['X2'] = np.array([x.flatten() for x in data_dict['X2']])
            data_dict['X2'] = np.concatenate(
                (data_dict['X2'], np.load(MODELPATHCNN.train_folder_landmarks_hog + '/hog_features.npy')),
                axis=1)
            data_dict['Y'] = np.load(MODELPATHCNN.train_folder_landmarks_hog + '/labels.npy')
        # data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        if use_hog_sliding_window_and_landmarks:
            data_dict['X'] = np.load(MODELPATHCNN.train_folder_landmarks_hog_sw + '/images.npy')
            data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            data_dict['X2'] = np.load(MODELPATHCNN.train_folder_landmarks_hog_sw + '/landmarks.npy')
            data_dict['X2'] = np.load(MODELPATHCNN.train_folder_landmarks_hog_sw + '/landmarks.npy')
            data_dict['X2'] = np.array([x.flatten() for x in data_dict['X2']])
            data_dict['X2'] = np.concatenate(
                (data_dict['X2'], np.load(MODELPATHCNN.train_folder_landmarks_hog_sw + '/hog_features.npy')),
                axis=1)
            data_dict['Y'] = np.load(MODELPATHCNN.train_folder_landmarks_hog_sw + '/labels.npy')

        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :, :]
            if use_landmarks and use_hog_and_landmarks:
                data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :]
            elif use_landmarks:
                data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to, :]

        if validation:
            # load validation set
            # validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
            # validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if use_landmarks:
                validation_dict['X'] = np.load(MODELPATHCNN.validation_folder_landmarks + '/images.npy')
                validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                validation_dict['X2'] = np.load(MODELPATHCNN.validation_folder_landmarks + '/landmarks.npy')
                validation_dict['Y'] = np.load(MODELPATHCNN.validation_folder_landmarks + '/labels.npy')
            if use_hog_and_landmarks:
                validation_dict['X'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog + '/images.npy')
                validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                validation_dict['X2'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog + '/landmarks.npy')
                validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
                validation_dict['X2'] = np.concatenate(
                    (validation_dict['X2'], np.load(MODELPATHCNN.validation_folder_landmarks_hog + '/hog_features.npy')), axis=1)
                validation_dict['Y'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog + '/labels.npy')
            # validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if use_hog_sliding_window_and_landmarks:
                validation_dict['X'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog_sw + '/images.npy')
                validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                validation_dict['X2'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog_sw + '/landmarks.npy')
                validation_dict['X2'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog_sw + '/landmarks.npy')
                validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
                validation_dict['X2'] = np.concatenate(
                    (
                    validation_dict['X2'], np.load(MODELPATHCNN.validation_folder_landmarks_hog_sw + '/hog_features.npy')),
                    axis=1)
                validation_dict['Y'] = np.load(MODELPATHCNN.validation_folder_landmarks_hog_sw + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :, :]
                if use_landmarks and use_hog_and_landmarks:
                    validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :]
                elif use_landmarks:
                    validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to, :]

        if test:
            # load test set
            # test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
            # test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if use_landmarks:
                test_dict['X'] = np.load(MODELPATHCNN.test_folder_landmarks + '/images.npy')
                test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                test_dict['X2'] = np.load(MODELPATHCNN.test_folder_landmarks + '/landmarks.npy')
                test_dict['Y'] = np.load(MODELPATHCNN.test_folder_landmarks + '/labels.npy')
            if use_hog_and_landmarks:
                test_dict['X'] = np.load(MODELPATHCNN.test_folder_landmarks_hog + '/images.npy')
                test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                test_dict['X2'] = np.load(MODELPATHCNN.test_folder_landmarks_hog + '/landmarks.npy')
                test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
                test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(MODELPATHCNN.test_folder_landmarks_hog + '/hog_features.npy')),
                                                 axis=1)
                test_dict['Y'] = np.load(MODELPATHCNN.test_folder_landmarks_hog + '/labels.npy')
            if use_hog_sliding_window_and_landmarks:
                test_dict['X'] = np.load(MODELPATHCNN.test_folder_landmarks_hog_sw + '/images.npy')
                test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
                test_dict['X2'] = np.load(MODELPATHCNN.test_folder_landmarks_hog_sw + '/landmarks.npy')
                test_dict['X2'] = np.load(MODELPATHCNN.test_folder_landmarks_hog_sw + '/landmarks.npy')
                test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
                test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(MODELPATHCNN.test_folder_landmarks_hog_sw + '/hog_features.npy')),
                                                 axis=1)
                test_dict['Y'] = np.load(MODELPATHCNN.test_folder_landmarks_hog_sw + '/labels.npy')
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :, :]
                if use_landmarks and use_hog_and_landmarks:
                    test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :]
                elif use_landmarks:
                    test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to, :]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else:
            return data_dict, validation_dict, test_dict
    else:
        print("Unknown dataset")
        exit()
