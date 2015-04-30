# Performs Logistic Regression on the data set.

from data import *
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

print("Running Logistic Regression...")

print("Performing 10-fold cross validation...")
clf = LogisticRegression()
scores = cross_validation.cross_val_score(clf, train_features, train_targets,
    cv = 10, scoring='log_loss')
print_validation_score(scores)

print("Predicting test data...")
clf.fit(train_features, train_targets)
test_predictions = clf.predict_proba(test_features)

write_submission(test_ids, test_predictions, 'logisticregression.csv')
