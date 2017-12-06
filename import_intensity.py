import cv2
import numpy as np
import os

dir_train_data = os.fsencode("training_data")

n = 10
# k = 5
total_data = 1000
no_train_data = 700
# data_width = 40
# data_height = 34
# c = 2.67
# gamma = 5.383

numbers = np.arange(n)

width = None
height = None

img_train = None
img_test = None

test_label = None
train_labels = None
train_labels_float = None


def preprocess_data(img_width, img_height):
    global img_train, img_test, train_labels, test_label, train_labels_float, height, width
    noOfTraining = 0
    noOfTesting = 0
    width = img_width
    height = img_height

    img_train = np.empty([no_train_data*n, img_width*img_height])
    img_test = np.empty([(total_data-no_train_data)*n, img_width*img_height])

    for digit_samples in numbers:
        i = 0
        for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
            print("Loading... %d%%\r" % ((digit_samples * 10) + i / 100), end="")
            img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
            kernel = np.ones((3,3),np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
            img = cv2.bitwise_not(img)
            if i < no_train_data:
                img_train[noOfTraining] = np.array(img).reshape(img_width * img_height)
                noOfTraining += 1
            else:
                img_test[noOfTesting] = np.array(img).reshape(img_width * img_height)
                noOfTesting += 1
            i += 1

    img_train = img_train.reshape(-1, img_width * img_height).astype(np.float32)
    img_test = img_test.reshape(-1, img_width * img_height).astype(np.float32)

    train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
    test_label = np.repeat(numbers, (total_data-no_train_data))[:, np.newaxis]

    train_labels_float = train_labels.astype(np.float32)


def trainKnn():
    print("# training knn...")
    knn = cv2.ml.KNearest_create()
    knn.train(img_train, cv2.ml.ROW_SAMPLE, train_labels_float)
    return knn


def doKnn(knn, test, k):
    knn_result = knn.findNearest(test, k)[1]
    return knn_result


def performKnn(k):
    print("# training...\r", end="")
    knn = cv2.ml.KNearest_create()
    knn.train(img_train, cv2.ml.ROW_SAMPLE, train_labels_float)

    print("# verifying...\r", end="")
    knn_result = knn.findNearest(img_test, k)[1]

    knn_matches = (knn_result == test_label)
    knn_correct = np.count_nonzero(knn_matches)
    knn_acc = knn_correct * 100.0 / knn_result.size
    return knn_acc


def performSVM(c, gamma):
    print("# training...\r", end="")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.train(img_train, cv2.ml.ROW_SAMPLE, train_labels)

    print("# verifying...\r", end="")
    svm_result = svm.predict(img_test)[1]

    svm_matches = (svm_result == test_label)
    svm_correct = np.count_nonzero(svm_matches)
    svm_acc = svm_correct*100.0/svm_result.size
    return svm_acc
