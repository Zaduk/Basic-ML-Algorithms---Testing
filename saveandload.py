# This script is a copy of linreg.py but it shows how we can use pickle to save and load a model.

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# read data from student-mat.csv, separated by ; instead of commas
data = pd.read_csv("student/student-mat.csv", sep=";")
# print("Before trimming:\n", data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print("After trimming:\n",data.head())

# predict is the label, what we're trying to predict
predict = "G3"

# return a new dataframe that does not contain G3; we will predict a value for G3 based on the training
x = np.array(data.drop([predict], 1))
# print(X)

# y is an array containing the G3 values
y = np.array(data[predict])
# print(y)

""" take all attributes and split them up into 4 arrays

x_train and y_train are sections of x and y respectively

x_test and y_test test the accuracy of the model that we will create by being initialized to \
contain only a small fraction of the test data, because if it had all of the testing data the \
model would be inaccurate since it would already know the prediction's results, since it has already \
seen all the data before!

"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""

# get the linear regression tool
linear = linear_model.LinearRegression()

# fit the data - implicitly makes a line of best fit
linear.fit(x_train, y_train)

# test model - returns a value that represents the accuracy of the model
accuracy = linear.score(x_test, y_test)

print(accuracy)

with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print out all attributes' best-fit line's 5 coefficients (the graph is in 5 dimensions!!)
print("Co: \n", linear.coef_)

# print out the best-fit line's intercept
print("Intercept: \n", linear.intercept_)

# now let's actually use this model to predict what grade a student will get!
predictions = linear.predict(x_test)

# print out all predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
