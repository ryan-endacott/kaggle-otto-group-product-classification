# Performs Naive Bayes on the data set.

from data import train_features, train_targets, validation_features, \
    validation_targets, print_accuracy
from sklearn.naive_bayes import GaussianNB

print("Running Naive Bayes...")
gnb = GaussianNB()
validation_predictions = \
    gnb.fit(train_features, train_targets).predict(validation_features)

print_accuracy(validation_predictions, validation_targets)
