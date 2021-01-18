from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm
from LBP_Skimage import *
from lbp_CV_hardCoded import *
import numpy as np
import argparse
import cv2
import os
import random
from RectangleCase import *
# Depending on library versions on your system, one of the following imports
from sklearn.model_selection import train_test_split
from ReadDataSets import *

path_to_dataset = r'data'
target_img_size = (32, 32)  # fix image size because classification algorithms THAT WE WILL USE HERE expect that

# We are going to fix the random seed to make our experiments reproducible
# since some algorithms use pseudorandom generators
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


def extract_lbp_histogram(imgName):
    img = GetTextureBlock(imgName)
    hist = lbp(img)
    return hist


def extract_features(img, feature_set='lbp-histogram'):
    return extract_lbp_histogram(img)


def load_dataset(case, feature_set='lbp-histogram'):
    features = []
    labels = [1, 1, 2, 2, 3, 3]

    for i in case:
        features.append(extract_features(i, feature_set))

    return features, labels


classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed)
}


# This function will test all our classifiers on a specific feature set
def run_experiment(testCase, case, feature_set):
    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    train_features, train_labels = load_dataset(case, feature_set)
    print('Finished loading dataset.')

    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)
        img = GetTextureBlock(testCase)
        hist = lbp(img)
        result = model.predict(hist)
        print(result)


Cases, testCases = LoadCases()
count = 0
for i in Cases:
    run_experiment(testCases[count], i, 'lbp-histogram')
