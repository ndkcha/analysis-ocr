import numpy as np
from matplotlib import pyplot as plt
import import_paper
import import_intensity
import import_hog

print("Intensity\n")
import_intensity.preprocess_data(20, 20)
i_k3 = import_intensity.performKnn(3)
i_k5 = import_intensity.performKnn(5)
i_k7 = import_intensity.performKnn(7)
print("performed kNN: [ok]")
i_svm = import_intensity.performSVM(2.67, 5.383)
print("performed SVM: [ok]")
print("k: 3 = ", i_k3, " | k: 5 = ", i_k5, " | k: 7 = ", i_k7, " | SVM = ", i_svm, "\n")

x = ["k:3", "k:5", "k:7", "SVM"]
y = [i_k3, i_k5, i_k7, i_svm]

plt.title("Research Paper: 34x40")
plt.subplot(1, 2, 1)
plt.bar(x, y)

plt.xlabel("k3: " + str(round(i_k3, 2)) + ", k5: " + str(round(i_k5, 2))
           + ", k7: " + str(round(i_k7, 2)) + ", svm: " + str(round(i_svm, 2)))
plt.ylabel("accuracy")

plt.legend()
plt.title("Intensity: 20x20")

print("Histogram of Gradient\n")
import_hog.preprocess_data(20)
hog_k3 = import_hog.performKnn(3)
hog_k5 = import_hog.performKnn(5)
hog_k7 = import_hog.performKnn(7)
print("performed kNN: [ok]")
hog_svm = import_hog.performSVM(2.67, 5.383)
print("performed SVM: [ok]")
print("k: 3 = ", hog_k3, " | k: 5 = ", hog_k5, " | k: 7 = ", hog_k7, " | SVM = ", hog_svm, "\n")

x = ["k:3", "k:5", "k:7", "SVM"]
y = [hog_k3, hog_k5, hog_k7, hog_svm]

plt.subplot(1, 2, 2)
plt.bar(x, y)

plt.xlabel("k3: " + str(round(hog_k3, 2)) + ", k5: " + str(round(hog_k5, 2))
           + ", k7: " + str(round(hog_k7, 2)) + ", svm: " + str(round(hog_svm, 2)))
plt.ylabel("accuracy")

plt.legend()
plt.title("HOG: 20x20")
plt.figure()

print("Research Paper\n")
tt_kNN_3 = np.zeros([10, 2])
tt_kNN_5 = np.zeros([10, 2])
tt_kNN_7 = np.zeros([10, 2])
tt_SVM = np.zeros([10, 2])

for i in np.arange(10):
    import_paper.preprocess_data((20, 20), (-1, 5, 5), 16, 40*(i+1))
    print(i+1, "\r")
    k = import_paper.performKnn(3)
    tt_kNN_3[i] = [import_paper.feature_space, k]
    k = import_paper.performKnn(5)
    tt_kNN_5[i] = [import_paper.feature_space, k]
    k = import_paper.performKnn(7)
    tt_kNN_7[i] = [import_paper.feature_space, k]
    s = import_paper.performSVM(2.67, 5.383)
    tt_SVM[i] = [import_paper.feature_space, s]

print("k", 3)
print(tt_kNN_3)
print("k", 5)
print(tt_kNN_5)
print("k", 7)
print(tt_kNN_7)
print("SVM")
print(tt_SVM)


x = tt_kNN_3[:, 0]
y = tt_kNN_3[:, 1]
plt.plot(x, y, label="kNN:3")
x = tt_kNN_5[:, 0]
y = tt_kNN_5[:, 1]
plt.plot(x, y, label="kNN:5")
x = tt_kNN_7[:, 0]
y = tt_kNN_7[:, 1]
plt.plot(x, y, label="kNN:7")
x = tt_SVM[:, 0]
y = tt_SVM[:, 1]
plt.plot(x, y, label="SVM")

plt.xlabel('feature space')
plt.ylabel('accuracy')

plt.legend()
plt.title("Research Paper: 20x20")

intensity = [('k3', i_k3), ('k7', i_k7), ('k5', i_k5), ('svm', i_svm)]
hog = [('k3', hog_k3), ('k5', hog_k5), ('k7', hog_k7), ('svm', hog_svm)]
paper = [('k3', np.amax(tt_kNN_3, axis=0)), ('k5', np.amax(tt_kNN_5, axis=0)),
         ('k7', np.amax(tt_kNN_7, axis=0)), ('svm', np.amax(tt_SVM, axis=0))]

b_i = max(intensity, key=lambda e: e[1])
b_hog = max(hog, key=lambda e: e[1])
b_paper = max(paper, key=lambda e: e[1][1])

print("Final Results (20x20):")
print("intensity", intensity)
print("hog", hog)
print("paper", paper)
print("Best:")
print("intensity", b_i[0], b_i[1])
print("hog", b_hog[0], b_hog[1])
print("paper", b_paper[0], b_paper[1])

plt.show()