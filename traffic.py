"""
TRAFFIC IMAGE CLASSIFICATION

@author: Hayk Stepanyan
"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH, IMG_HEIGHT = 30, 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

def load_data(data_dir):
    """
    Args:
        data_dir - directory
    Return:
        tuple (images, labels), where images is the list of images and 
        labels is a list of integers representing corresponding labels
    """
    images, labels = [], []
    for category in range(43):
        for image_name in os.listdir(os.path.join(data_dir, str(category))):
            image = cv2.imread(os.path.join(data_dir, str(category), image_name))
            images.append(cv2.resize(image, (30, 30)))
            labels.append(category)
    return images, labels

def load_model():
    """
    Return:
        Trained CNN
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (4, 4), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.Conv2D(
            64, (4, 4), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            10, (3, 3)
        ),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            10, (3, 3)
        ),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if len(sys.argv) not in [2, 3]:
    sys.exit("Usage: python traffic.py data_directory [model.h5]")

images, labels = load_data("gtsrb")

# Split dataset into training and test
labels = tf.keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images), np.array(labels), test_size=TEST_SIZE
)


model = load_model()
model.fit(x_train, y_train, epochs=EPOCHS)
model.evaluate(x_test,  y_test, verbose=2)

# Save model to file
if len(sys.argv) == 3:
    filename = sys.argv[2]
    model.save(filename)
    print("Model saved to {}".format(filename))
