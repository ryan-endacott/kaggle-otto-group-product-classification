# Performs Naive Bayes on the data set.

from data import *
from sklearn.naive_bayes import GaussianNB

print("Running Naive Bayes...")
gnb = GaussianNB()
gnb.fit(train_features, train_targets)
validation_predictions = gnb.predict(validation_features)

print_accuracy(validation_predictions, validation_targets)

test_predictions = gnb.predict(test_features)

write_submission(test_data['id'], test_predictions, 'naivebayes.csv')
