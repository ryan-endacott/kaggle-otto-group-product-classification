# Gets statistics about PCA variance on the dataset.

from data import *
from sklearn.decomposition import PCA

# Note: Total number of features is 93.
pca = PCA(n_components = 93)
pca.fit(train_features)
variance_ratio = pca.explained_variance_ratio_

def print_pca_explained_variance(n_components):
   print("The variance explained by {0} components is {1}.\n"
           .format(n_components, sum(variance_ratio[0:n_components])))

print_pca_explained_variance(93)
print_pca_explained_variance(92)
print_pca_explained_variance(90)
print_pca_explained_variance(50)
print_pca_explained_variance(30)
print_pca_explained_variance(10)
print_pca_explained_variance(5)



