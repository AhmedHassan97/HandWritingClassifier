from sklearn import svm
from Imports import *
from lbp_CV_hardCoded import *
from RectangleCase import *
from ReadHFdataset import *
from LBP_Skimage import *
import time

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


def extract_lbp_histogram(imgName):
    blocks = GetTextureBlock(imgName)
    listofHists = []

    for i in blocks:
        hist = lbp(i)
        listofHists.append(hist)
    return listofHists


def extract_features(img, feature_set='lbp-histogram'):
    return extract_lbp_histogram(img)


def load_dataset(case, feature_set='lbp-histogram'):
    features = []
    labels = []
    count = 0
    writer = 1
    for i in case:
        if count == 2:
            count=0
            writer += 1
        count += 1
        errorCounter = 0
        try:
            blocks = extract_features(i, feature_set)
            for j in blocks:
                if j.shape == (256,):
                    features.append(j)
                    labels.append(writer)
                    errorCounter += 1
        except:
            print("exception")
            pass

    return features, labels


classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed, dual=True, max_iter=1000000, C=300)
}


# This function will test all our classifiers on a specific feature set
def run_experiment(testCase, case, feature_set):
    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')

    CasesImage = []
    for element in case:
        CasesImage.append(io.imread(element))
    testCaseImage = io.imread(testCase)

    start = time.time()

    train_features, train_labels = load_dataset(CasesImage, feature_set)
    print('Finished loading dataset.')
    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)

        blocks = GetTextureBlock(testCaseImage)
        listofHists = []
        for i in blocks:
            hist = lbp(i)
            if hist.shape == (256,):
                listofHists.append(hist)
        result = []
        for i in listofHists:
            result.append(model.predict(np.reshape(i, (1, -1)))[0])

        result = np.asarray(result)
        writer = [np.sum(result == 1), np.sum(result == 2), np.sum(result == 3)]

        print(result, "SVM Result")
        end = time.time()
        f = open("results.txt", "a")
        f.write(str(np.argmax(writer) + 1)+"\n")
        f.close()
        f = open("time.txt", "a")
        f.write(str(round((end - start),2))+"\n")
        f.close()

        print("Time:", str(round((end - start),2)))


Cases, testCases = LoadCases()
count = 0
trueClassification = 0
for i in Cases:
    print(count, "case Number")
    run_experiment(testCases[count], i, 'lbp-histogram')
    count += 1

