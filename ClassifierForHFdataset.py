from sklearn import svm
from Imports import *
from lbp_CV_hardCoded import *
from RectangleCase import *
from ReadHFdataset import *

path_to_dataset = r'data'
target_img_size = (32, 32)  # fix image size because classification algorithms THAT WE WILL USE HERE expect that

# We are going to fix the random seed to make our experiments reproducible
# since some algorithms use pseudorandom generators
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


def extract_lbp_histogram(imgName):
    blocks = GetTextureBlock(imgName)
    listofHists =np.array([])

    for i in blocks:
        hist = lbp(i)
        listofHists = np.concatenate((listofHists, hist[0]))
    listofHists=listofHists/listofHists.sum() + 0.00000000000000000001
    # listofHists = listofHists.reshape(1,-1)
    print(listofHists.shape)
    print(listofHists,"dataset")
    # for i in range(9):
    #     copy=np.copy(listofHists[i][0])
    #     copy=np.reshape(copy,(256,1))
    #     print(copy.shape)
    # np.array(listofHists[0])
    # # np.mean()
    # print([listofHists[0][0].shape])
    return listofHists


def extract_features(img, feature_set='lbp-histogram'):
    return extract_lbp_histogram(img)


def load_dataset(case, feature_set='lbp-histogram'):
    features = []
    labels = [1, 1, 2, 2, 3, 3]

    for i in case:
        features.append(extract_features(i, feature_set))

    return features, labels


classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed,dual=True,max_iter=1000000,C=100)
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

        blocks = GetTextureBlock(testCase)
        listofHists = np.array([])
        for i in blocks:
            hist = lbp(i)
            listofHists = np.concatenate((listofHists, hist[0]))
        listofHists = listofHists.reshape(1, -1)
        listofHists = listofHists / listofHists.sum() + 0.00000000000000000001
        print(listofHists.shape)
        print(listofHists)
        result = model.predict(listofHists)
        print(result, "SVM Result")


Cases, testCases = LoadCases()
count = 0
for i in Cases:
    print(testCases[0])
    run_experiment(testCases[count], i, 'lbp-histogram')
    count += 1
    break
