# Performs Nearest Neighbors on the data set.

from data import *
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

n_components = 10
n_neighbors = 5
print(("Running KNeighborsClassifier with n_neighbors = {0} on PCAed data " +
        "with {1} components...").format(n_neighbors, n_components))

pca = PCA(n_components=n_components)
pca.fit(train_features)

train_features_reduced = pca.transform(train_features)
test_features_reduced = pca.transform(test_features)

print("Performing 10-fold cross validation...")
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
scores = cross_validation.cross_val_score(clf, train_features, train_targets,
    cv = 10, scoring='log_loss')
print_validation_score(scores)

print("Predicting test data...")
clf.fit(train_features_reduced, train_targets)
test_predictions = clf.predict_proba(test_features_reduced)

write_submission(test_ids, test_predictions, 'nearestneighbors.csv')
