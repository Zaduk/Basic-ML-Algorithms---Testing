import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()

# convert the non-numeric categories into numeric categories – returns a numpy array
buying = le.fit_transform((data["buying"]))
maint = le.fit_transform((data["maint"]))
door = le.fit_transform((data["door"]))
persons = le.fit_transform((data["persons"]))
lug_boot = le.fit_transform((data["lug_boot"]))
safety = le.fit_transform((data["safety"]))
cls = le.fit_transform((data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # zip creates tuples
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# choose k = 5
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print("Accuracy rate: ", accuracy)
# Note: for this model, K = 9 yields greater accuracy than K = 11 --> greater K doesn't
# always mean greater accuracy!


predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", predicted[x], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

    '''Now let's return the K-neighbors of a point: their indices in the dataset and
    and their distances to each query point. The method takes in a 2D array (hence the outer brackets),
    K, and a bool that decides whether or not to return the distances.'''
    n = model.kneighbors([x_test[x]], 9, True)
    print("n: ", n)