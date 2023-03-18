from sklearn.decomposition import FastICA
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.stats as stats

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
        c=sample_set.labels.astype('|S1').view(np.uint8),
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


def graph_analysis(task: Task):
    sample_set = data_utils.get_all_samples(task)
    num_features = sample_set.samples.shape[1]
    all_kurtosis_scores = []
    components_to_try = np.arange(1, num_features + 1)
    for num_components in components_to_try:
        kurtosis_scores = []
        ica, _ = transform(sample_set, num_components)
        for component in ica.components_:
            abs_kurtosis = np.abs(stats.kurtosis(component))
            kurtosis_scores.append(abs_kurtosis)
        kurtosis_scores = np.pad(kurtosis_scores, (0, num_features - num_components))
        all_kurtosis_scores.append(kurtosis_scores)

    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = np.tile(components_to_try, num_features)
    y3 = np.repeat(components_to_try, num_features)
    z3 = np.zeros(np.square(num_features))

    dx = .25 * np.ones(np.square(num_features))
    dy = .25 * np.ones(np.square(num_features))
    dz = np.array(all_kurtosis_scores).flatten()
    print(all_kurtosis_scores)

    ax1.bar3d(x3, y3, z3, dx, dy, dz, color='w')

    ax1.set_xlabel('projection')
    ax1.set_ylabel('# of components')
    ax1.set_zlabel('kurtosis')

    plt.show()


def transform(sample_set: SampleSet, num_components):
    ica = FastICA(n_components=num_components,
                  algorithm='parallel',
                  whiten='unit-variance',
                  fun='logcosh',
                  max_iter=2000,
                  whiten_solver='eigh',
                  random_state=0)
    ica.fit(sample_set.samples)
    return ica, ica.transform(sample_set.samples)
