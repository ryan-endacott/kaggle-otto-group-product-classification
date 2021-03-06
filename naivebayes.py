# Performs Naive Bayes on the data set.

from data import *
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

print("Running GaussianNB...")

print("Performing 10-fold cross validation...")
clf = GaussianNB()
scores = cross_validation.cross_val_score(clf, train_features, train_targets,
    cv = 10, scoring='log_loss')
print_validation_score(scores)

print("Predicting test data...")
clf.fit(train_features, train_targets)
test_predictions = clf.predict_proba(test_features)

write_submission(test_ids, test_predictions, 'naivebayes.csv')
