"""Digit classifier function, using saved CNN model."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.draw import line


def _preprocess(digit_data: np.ndarray) -> np.array:
    """Preprocess the 3D digit data for classification.

    Args:
        digit_data (np.ndarray): coordinates of the gesture.

    Returns:
        np.array: preprocessed data.
    """
    # check if the shape matches (3, _)
    if digit_data.shape[0] != 3:
        digit_data=digit_data.T

    # normalize coordinates
    x_new = (digit_data[0] - digit_data[0].min()) / (digit_data[0].max() - digit_data[0].min())
    y_new = (digit_data[1] - digit_data[1].min()) / (digit_data[1].max() - digit_data[1].min())

    # generalize size to 28x28
    grid_size = 28
    x_final = (x_new * (grid_size - 1)).astype(int)
    y_final = (y_new * (grid_size - 1)).astype(int)

    # create an empty matrix and connect the dots
    digit_conn_mat = np.zeros((grid_size, grid_size))
    for i in range(len(x_final) - 1):
        rr, cc = line(y_final[i], x_final[i], y_final[i + 1], x_final[i + 1])
        digit_conn_mat[rr, cc] = 1

    # flip vertically to correct orientation
    digit_conn_mat = digit_conn_mat[::-1]

    # reshape for CNN
    X = np.array(digit_conn_mat).reshape(-1, 28, 28, 1)

    return X


def digit_classify(digit: np.ndarray) -> int:
    """Classify the digit using CNN.

    Args:
        digit (np.ndarray): 3D coordinates of the digit.

    Returns:
        int: predicted digit.
    """
    # process digit gesture data -> (28, 28, 1)
    X = _preprocess(digit)

    # load trained CNN model
    DC = tf.keras.models.load_model('digit_classifier.keras', compile=False)

    # classify digit
    prediction = DC.predict(X)

    return np.argmax(prediction)
