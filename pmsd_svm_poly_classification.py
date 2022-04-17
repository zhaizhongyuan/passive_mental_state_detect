import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.utils import shuffle
import time

# Load processed features
data_dir = "./Data/"

# features (every row is a data point)
X_focused = pd.read_pickle(data_dir + 'focused_rd.pkl').transpose()
print("Dimensionality-reduced focused features loaded. {}".format(X_focused.shape))
X_unfocused = pd.read_pickle(data_dir + 'unfocused_rd.pkl').transpose()
print("Dimensionality-reduced unfocused features loaded. {}".format(X_unfocused.shape))
X_drowsy = pd.read_pickle(data_dir + 'drowsy_rd.pkl').transpose()
print("Dimensionality-reduced drowsy features loaded. {}".format(X_drowsy.shape))

# labels
y_focused = np.full((X_focused.shape[0], 1), 'focused')
y_unfocused = np.full((X_unfocused.shape[0], 1), 'unfocused')
y_drowsy = np.full((X_drowsy.shape[0], 1), 'drowsy')

print("focused:", X_focused.shape, y_focused.shape)
print("unfocused:", X_unfocused.shape, y_unfocused.shape)
print("drowsy:", X_drowsy.shape, y_drowsy.shape)

# Hyper-parameter
test_size = 0.2     # test data size
random_state = 42   # random generator number
cv = 5  # cross validation generator

# combine data and labels from different classes
X_data = np.concatenate((X_focused, X_unfocused, X_drowsy))
y_data = np.concatenate((y_focused, y_unfocused, y_drowsy))
# shuffle the data
X_data, y_data = shuffle(X_data, y_data)

print(X_data.shape, y_data.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=test_size, random_state=random_state, shuffle=True)

start = time.time()

# polynomial kernel
poly = svm.SVC(kernel='poly', degree=3, C=1, random_state=42, decision_function_shape='ovr')
# cross validation
scores = cross_val_score(poly, X_data, y_data.ravel(), scoring='accuracy')

end = time.time()

print("time:", end - start, "seconds")

print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
