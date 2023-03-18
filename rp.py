from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import matplotlib.pyplot as plt

from data_utils import Task, SampleSet
import data_utils

### Docs used:
# https://scikit-learn.org/stable/modules/random_projection.html
# https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection


def plot_3d(task: Task):
    sample_set = data_utils.get_all_samples(task)
    pca, transformed_data = transform(sample_set, 3)

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    ax.scatter(
        transformed_data[:, 0],
        transformed_data[:, 1],
        transformed_data[:, 2],
        c=sample_set.labels.astype('|S1').view(np.uint8),
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three RCA projections")
    ax.set_xlabel("1st projection")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd projection")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd projection")
    ax.zaxis.set_ticklabels([])
    plt.show()


def transform(sample_set: SampleSet, num_components):
    rca = GaussianRandomProjection(n_components=num_components, random_state=0)
    rca.fit(sample_set.samples)
    return rca, rca.transform(sample_set.samples)
