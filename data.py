# Loads the data.

import pandas as pd
import numpy as np
import scipy as sp

# For reproducibility, although I'm not sure if it's needed for any of the
# scikit learn algorithms or if it even affects them.
np.random.seed(1337)

train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

# Set up the features and targets.
train_features = train_data
train_features = train_features.drop('target', 1)
train_features = train_features.drop('id', 1)

test_features = test_data.drop('id', 1)
test_ids = test_data['id']

train_targets = train_data['target']

def print_validation_score(scores):
    print("Cross validation logloss score: {0} +- {1}"
            .format(-scores.mean(), scores.std() * 2))

def write_submission(ids, predictions, filename):
    print("Writing submission to submissions/{0}...".format(filename))

    # Build the whole file contents to be written for efficiency.
    pred_str_arr = []
    for idx, prediction in enumerate(predictions):
        pred_str_arr.append(
                "%d,%s\n" % (ids[idx], ','.join(str(p) for p in prediction)))

    with open('submissions/' + filename, 'w') as f:
        f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7" +
                ",Class_8,Class_9\n")
        f.write(''.join(pred_str_arr))
    print("Done writing submission.")
