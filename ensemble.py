# Performs an ensemble of the random forest, svc, and neural network
# submissions.

import pandas as pd
import numpy as np
from data import write_submission, test_ids

submission_csvs = [
    'submissions/randomforest.csv',
    'submissions/keras-otto-novalidation.csv',
    'submissions/svm.csv']

predictions = [pd.read_csv(csv).drop('id', 1).values for csv in submission_csvs]

# Weight predictions
rf_pred = predictions[0] * 2
keras_pred = predictions[1] * 3
svm_pred = predictions[2] * 2

ensemble_prediction = reduce(np.add, [rf_pred, keras_pred, svm_pred])

write_submission(test_ids, ensemble_prediction, 'ensemble.csv')
