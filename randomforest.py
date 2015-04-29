# Performs Naive Bayes on the data set.

from data import *
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

n_estimators = 100
print("Running Random Forest with n_estimators = {0}...".format(n_estimators))

print("Performing 10-fold cross validation...")
clf = RandomForestClassifier(n_estimators=n_estimators)
scores = cross_validation.cross_val_score(clf, train_features, train_targets,
    cv = 10, scoring='log_loss')
print_validation_score(scores)

print("Predicting test data...")
clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(train_features, train_targets)
test_predictions = clf.predict_proba(test_features)

write_submission(test_ids, test_predictions, 'randomforest.csv')
