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

