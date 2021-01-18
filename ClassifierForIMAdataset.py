from sklearn import svm
from sklearn.model_selection import train_test_split

from Imports import *
from lbp_CV_hardCoded import *
from RectangleCase import *

from ReadIMAdatset import *

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


def load_dataset(feature_set='lbp-histogram'):
    features = []
    imageDir, labels = readIMAdata()

    for i in imageDir:
        features.append(extract_features(i, feature_set))

    return features, labels


classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed)
}


# This function will test all our classifiers on a specific feature set
def run_experiment(feature_set):
    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(feature_set)
    print('Finished loading dataset.')

    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)

    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)

        # Test the model on images it hasn't seen before
        accuracy = model.score(test_features, test_labels)

        print(model_name, 'accuracy:', accuracy * 100, '%')


run_experiment('lbp-histogram')

# Cases, testCases = LoadCases()
# count = 0
# for i in Cases:
#     print(testCases[0])
#     count += 1
