# Performs Support Vector Machine classification on the data set.

from data import *
from sklearn import cross_validation
from sklearn.svm import SVC

print("Running SVC...")

print("Performing 10-fold cross validation...")
clf = SVC(probability=True)
scores = cross_validation.cross_val_score(clf, train_features, train_targets,
    cv = 10, scoring='log_loss')
print_validation_score(scores)

print("Predicting test data...")
clf.fit(train_features, train_targets)
test_predictions = clf.predict_proba(test_features)

write_submission(test_ids, test_predictions, 'svm.csv')
