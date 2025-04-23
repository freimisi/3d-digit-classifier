"""Sample script to test the digit_classify function on a single digit from the training data."""

import os
import numpy as np
import pandas as pd
from digit_classifier import digit_classify


if __name__ == '__main__':
    folder_path = 'digits_3d/training_data/'
    file_name = 'stroke_7_0006.csv'
    to_load = os.path.join(folder_path, file_name)
    digit = pd.read_csv(to_load, header=None)  # load digit
    digit = digit.to_numpy()                   # convert to numpy array
    prediction = digit_classify(digit)         # classify digit

    print(f"pred: {prediction}\nactual: {int(file_name[7])}")
