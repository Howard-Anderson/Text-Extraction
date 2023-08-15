"""
                Text Extraction from Images.

        Project: Text Extraction from Images.

        Author: Howard Anderson.

        Date: 19/4/2023.

        Filename: build.py

        Description: Script to Build the Model.

        Dataset: EMNIST
"""


import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from emnist import extract_training_samples
from emnist import extract_test_samples

def read_data():
    # Reading in the Data.
    dtrain, dlabels = extract_training_samples("digits")
    dtest, dtest_labels = extract_test_samples("digits")
    ltrain, llabels = extract_training_samples("letters")
    ltest, ltest_labels = extract_test_samples("letters")
    llabels, ltest_labels = llabels + 9, ltest_labels + 9

    # Combining Letters and Digits Datasets.
    train = np.append(dtrain,ltrain, axis = 0)
    train_labels = np.append(dlabels, llabels, axis = 0)
    test = np.append(dtest, ltest, axis = 0)
    test_labels = np.append(dtest_labels, ltest_labels, axis = 0)

    return train.reshape(-1,28,28,1), train_labels, test.reshape(-1,28,28,1), test_labels


x_train, y_train, x_test, y_test = read_data()

scaled_x_train, scaled_x_test = x_train / 255.0, x_test / 255.0

print(f"\n\nShape: Images: {x_train.shape}, Labels: {y_train.shape}")

print(f"\n\nLabel for 200000: {y_train[360000]}")


# Model Architecture.
classifier = Sequential()
classifier.add(Conv2D(28, (3,3), input_shape = (28,28,1), activation = "relu"))
classifier.add(MaxPooling2D((2,2)))
classifier.add(Conv2D(56, (3,3), activation = "relu"))
classifier.add(MaxPooling2D((2,2)))
classifier.add(Conv2D(56, (3,3), activation = "relu"))
classifier.add(Flatten())
classifier.add(Dense(56, activation = "relu"))
classifier.add(Dense(36))

# Model Summary.
classifier.summary()

classifier.compile(optimizer = "adam",
                   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                   metrics = ["accuracy"]
                   )

fitted_classifier = classifier.fit(scaled_x_train, y_train, epochs = 10, validation_data = (scaled_x_test, y_test))

loss, accuracy = classifier.evaluate(scaled_x_test, y_test, verbose = 2)

print(f"\nTest-Loss: {loss}, Test-Accuracy: {accuracy}\n\n")

