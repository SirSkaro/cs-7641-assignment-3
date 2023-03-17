from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt

from data_utils import Task, SampleSet
import data_utils


### Docs used:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA

def plot_3d(task: Task):
    sample_set = data_utils.get_all_samples(task)
    pca, transformed_data = transform(sample_set, 3)

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    ax.scatter(
        transformed_data[:, 0],
        transformed_data[:, 1],
        transformed_data[:, 2],
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three ICA projections")
    ax.set_xlabel("1st projection")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd projection")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd projection")
    ax.zaxis.set_ticklabels([])
    plt.show()


def transform(sample_set: SampleSet, num_components):
    ica = FastICA(n_components=num_components,
                  algorithm='parallel',
                  whiten='warn',
                  fun='logcosh',
                  max_iter=100,
                  whiten_solver='eigh')
    ica.fit(sample_set.samples)
    return ica, ica.transform(sample_set.samples)
