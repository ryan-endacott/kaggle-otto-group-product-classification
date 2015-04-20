# Loads the data.

import pandas as pd
import numpy as np

kaggle_train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

# Set up the features and targets.
kaggle_train_features = kaggle_train_data
kaggle_train_features = kaggle_train_features.drop('target', 1)
kaggle_train_features = kaggle_train_features.drop('id', 1)

test_features = test_data.drop('id', 1)
test_ids = test_data['id']

kaggle_train_targets = kaggle_train_data['target']

# Split kaggle training data into test and validation with a split of
# 80% test, 20% validation.
np.random.seed(1337)
split_value = int(len(kaggle_train_features) * .8)
rows = np.random.choice(kaggle_train_features.index.values, split_value,
        replace=False)
train_features = kaggle_train_features.ix[rows]
validation_features = kaggle_train_features.drop(rows)
train_targets = kaggle_train_targets.ix[rows]
validation_targets = kaggle_train_targets.drop(rows)

def print_accuracy(predictions, targets):
    total = len(predictions)
    if total != len(targets):
        print "Error: Prediction and target dimensions don't match."
        return
    num_correct = (predictions == targets).sum()
    percent_correct = (float(num_correct) / total) * 100
    print("Got {0}% accuracy with {1} correct predictions out of {2}."
            .format(percent_correct, num_correct, total))
    return

def write_submission(ids, predictions, filename):
    print("Writing submission to {0}...".format(filename))
    with open('submissions/' + filename, 'w') as f:
        f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7" +
            ",Class_8,Class_9\n")
        for idx, prediction in enumerate(predictions):
            results = [0] * 9
            prediction_int = int(prediction[-1])
            results[prediction_int - 1] = 1
            f.write("{0},{1}\n"
                    .format(ids[idx], ','.join(str(res) for res in results)))

# Reseed random so it's not based on the same seed anymore.
np.random.seed()
