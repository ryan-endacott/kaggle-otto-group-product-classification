# Loads the data.

import pandas as pd

kaggle_train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

# Set up the features and targets.
kaggle_train_features = kaggle_train_data
kaggle_train_features = kaggle_train_features.drop('target', 1)
kaggle_train_features = kaggle_train_features.drop('id', 1)

test_features = test_data.drop('id', 1)

kaggle_train_targets = kaggle_train_data['target']

# Split kaggle training data into test and validation with a split of
# 80% test, 20% validation.
split_index = int(len(kaggle_train_features) * .8)
train_features = kaggle_train_features[:split_index]
validation_features = kaggle_train_features[split_index:]
train_targets = kaggle_train_targets[:split_index]
validation_targets = kaggle_train_targets[split_index:]

def print_accuracy(predictions, targets):
    total = len(predictions)
    if total != len(targets):
        print "Error: Prediction and target dimensions don't match."
        return
    num_correct = (predictions == targets).sum()
    percent_correct = (float(num_correct) / total) * 100
    print("Got {0}% accuracy with {1} correct predictions out of {2}."
            .format(percent_correct, num_correct, total))

