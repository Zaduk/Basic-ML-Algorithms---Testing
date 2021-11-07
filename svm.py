import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#  0.2 for more data to test with

# print(x_train, y_train)

# you can tweak the parameter for better or worse accuracy
# C is the soft margin. Default = 1x number points allowed in soft margin
clf = svm.SVC(kernel="linear", C=4)
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)

print(accuracy)

# Would using KNN yield a better accuracy?
# Change svm.SVC(kernel="linear", C=4) to KNeighborsClassifier(n_neighbours=10) and compare the accuracy!

# SVM has a better overall accuracy over KNN.


