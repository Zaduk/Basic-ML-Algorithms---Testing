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

print(x_train, y_test)
