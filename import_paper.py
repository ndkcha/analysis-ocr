import numpy as np
import cv2
import os
import math

dir_train_data = os.fsencode("training_data")

n = 10
total_data = 1000
no_train_data = 700
# k = 5
# c = 2.67
# gamma = 5.383

numbers = np.arange(n)
img_train = None
img_test = None
train_labels = None
train_labels_float = None
test_label = None
sub_mat_shape = None
sub_mat_size = 0
fft_n = None

feature_space = 0


def preprocess_data(img_size, sub_matrix_shape, sub_matrix_size, fft_no):
    global img_train, img_test, train_labels, train_labels_float, test_label, feature_space, sub_mat_shape, fft_n, sub_mat_size
    noOfTraining = 0
    noOfTesting = 0
    feature_space = (sub_matrix_size * 3) + fft_no
    sub_mat_shape = sub_matrix_shape
    sub_mat_size = sub_matrix_size
    fft_n = fft_no

    img_train = np.empty([no_train_data * n, feature_space])
    img_test = np.empty([(total_data - no_train_data) * n, feature_space])

    for digit_samples in numbers:
        i = 0
        for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
            print("# Loading... %d%%\r" % ((digit_samples * 10) + i / 100), end="")
            ff = 0
            sf = sub_matrix_size
            tf = sub_matrix_size * 2
            feature_vector = np.zeros(feature_space)
            img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
            img = cv2.bitwise_not(img)

            div_img = np.array(img).reshape(sub_matrix_shape)
            for sub_matrix in div_img:
                zeros= np.where(sub_matrix != 0)
                feature_vector[ff] = zeros[0].size
                if feature_vector[ff] == 0:
                    continue
                co_zeros = np.transpose(zeros)
                mean_distance = 0.0
                mean_angle = 0.0
                for coordinates in co_zeros:
                    mean_distance += ((coordinates[0]**2) + (coordinates[1]**2))**0.5
                    mean_angle += coordinates[0] == 0 and 90 or math.degrees(math.atan(coordinates[1]/coordinates[0]))
                feature_vector[sf] = mean_distance/float(feature_vector[ff])
                feature_vector[tf] = mean_angle/float(feature_vector[ff])
                ff += 1
                sf += 1
                tf += 1

            ff_img = np.fft.fft(np.array(img.reshape(-1)))
            fft_img = ((ff_img.real ** 2) + (ff_img.imag ** 2)) ** 0.5
            feature_vector[(sub_matrix_size*3):] = fft_img[:fft_no]

            if i < no_train_data:
                img_train[noOfTraining] = feature_vector
                noOfTraining += 1
            else:
                img_test[noOfTesting] = feature_vector
                noOfTesting += 1
            i += 1

    img_train = img_train.reshape(-1, feature_space).astype(np.float32)
    img_test = img_test.reshape(-1, feature_space).astype(np.float32)

    train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
    test_label = np.repeat(numbers, (total_data - no_train_data))[:, np.newaxis]

    train_labels_float = train_labels.astype(np.float32)


def preprocess_test(img):
    ff = 0
    sf = sub_mat_size
    tf = sub_mat_size * 2
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)

    feature_vector = np.zeros(feature_space)

    div_img = np.array(img).reshape(sub_mat_shape)
    for sub_matrix in div_img:
        zeros = np.where(sub_matrix != 0)
        feature_vector[ff] = zeros[0].size
        if feature_vector[ff] == 0:
            continue
        co_zeros = np.transpose(zeros)
        mean_distance = 0.0
        mean_angle = 0.0
        for coordinates in co_zeros:
            mean_distance += ((coordinates[0] ** 2) + (coordinates[1] ** 2)) ** 0.5
            mean_angle += coordinates[0] == 0 and 90 or math.degrees(math.atan(coordinates[1] / coordinates[0]))
        feature_vector[sf] = mean_distance / float(feature_vector[ff])
        feature_vector[tf] = mean_angle / float(feature_vector[ff])
        ff += 1
        sf += 1
        tf += 1

    ff_img = np.fft.fft(np.array(img.reshape(-1)))
    fft_img = ((ff_img.real ** 2) + (ff_img.imag ** 2)) ** 0.5
    feature_vector[(sub_mat_size * 3):] = fft_img[:fft_n]
    return feature_vector


def trainKnn():
    print("# training...\r", end="")
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
    ret, knn_result, neighbour, dist = knn.findNearest(img_test, k)
    knn_matches = (knn_result == test_label)
    knn_correct = np.count_nonzero(knn_matches)

    knn_acc = knn_correct*100.0/knn_result.size
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
