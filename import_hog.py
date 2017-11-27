import cv2
import numpy as np
import os

dir_train_data = os.fsencode("training_data")

n = 10
total_data = 1000
no_train_data = 700
# k = 5
# c = 2.67
# gamma = 5.383
img_size = None
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
bin_n = 16

numbers = np.arange(n)

img_train = np.empty([no_train_data*n, 64])
img_test = np.empty([(total_data-no_train_data)*n, 64])

test_label = None
train_labels = None
train_labels_float = None


def deskewMoments(image):
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img_size[0] * skew], [0, 1, 0]])
    image = cv2.warpAffine(image, M, img_size, flags=affine_flags)
    return image


def deskewAngle(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image


def hog(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def preprocess_data(dim):
    global img_size, img_train, img_test, train_labels, test_label, train_labels_float
    noOfTraining = 0
    noOfTesting = 0
    img_size = (dim, dim)

    for digit_samples in numbers:
        i = 0
        for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
            print("Loading digits # %d%%\r" % ((digit_samples * 10) + i / 100), end="")
            img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
            img = cv2.resize(img, img_size)
            img = cv2.bitwise_not(img)
            d_img = deskewMoments(img)
            h_img = hog(d_img)
            if i < no_train_data:
                img_train[noOfTraining] = np.array(h_img).reshape(64)
                noOfTraining += 1
            else:
                img_test[noOfTesting] = np.array(h_img).reshape(64)
                noOfTesting += 1
            i += 1

    img_train = img_train.reshape(-1, 64).astype(np.float32)
    img_test = img_test.reshape(-1, 64).astype(np.float32)

    train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
    test_label = np.repeat(numbers, (total_data-no_train_data))[:, np.newaxis]

    train_labels_float = train_labels.astype(np.float32)


def preprocess_test(img):
    img = cv2.bitwise_not(img)
    d_img = deskewMoments(img)
    h_img = hog(d_img)
    return h_img


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
