from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from data_utils import Task, SampleSet
import data_utils

### Docs used:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca


def transform(sample_set: SampleSet, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(sample_set.samples)
    return pca, pca.transform(sample_set.samples)
