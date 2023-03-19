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
# https://subscription.packtpub.com/book/big-data-/9781849513265/7/ch07lvl1sec78/creating-a-3d-bar-plot
# https://pythonprogramming.net/3d-bar-chart-matplotlib-tutorial/
# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

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


def graph_analysis(sample_set: SampleSet):
    num_features = sample_set.samples.shape[1]
    all_kurtosis_scores = []
    components_to_try = np.arange(1, num_features + 1)
    for num_components in components_to_try:
        kurtosis_scores = []
        ica, transformed_data = transform(sample_set, num_components)
        for component in range(num_components):
            abs_kurtosis = np.abs(stats.kurtosis(transformed_data[:, component]))
            kurtosis_scores.append(abs_kurtosis)
        kurtosis_scores = np.pad(kurtosis_scores, (0, num_features - num_components))
        all_kurtosis_scores.append(kurtosis_scores)

    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = np.tile(components_to_try, num_features)
    y3 = np.repeat(components_to_try, num_features)
    z3 = np.zeros(np.square(num_features))

    dx = .5 * np.ones(np.square(num_features))
    dy = .5 * np.ones(np.square(num_features))
    dz = np.array(all_kurtosis_scores).flatten()

    colors = np.repeat(np.random.rand(num_features, 3), num_features, axis=0)

    ax1.bar3d(x3, y3, z3, dx, dy, dz, color=colors)

    ax1.set_xlabel('projection')
    ax1.set_ylabel('# of sources')
    ax1.set_zlabel('kurtosis')

    plt.show()
    return all_kurtosis_scores


def choose_num_components(sample_set: SampleSet):
    num_features = sample_set.samples.shape[1]
    all_kurtosis_scores = []
    components_to_try = np.arange(1, num_features + 1)
    for num_components in components_to_try:
        kurtosis_scores = []
        ica, transformed_data = transform(sample_set, num_components, True)
        for component in range(num_components):
            abs_kurtosis = np.abs(stats.kurtosis(transformed_data[:, component]))
            kurtosis_scores.append(abs_kurtosis)
        kurtosis_scores = np.pad(kurtosis_scores, (0, num_features - num_components))
        all_kurtosis_scores.append(kurtosis_scores)

    all_kurtosis_scores = np.array(all_kurtosis_scores)
    all_kurtosis_scores[all_kurtosis_scores == 0] = np.nan
    averages = np.nanmean(all_kurtosis_scores, axis=1)
    return averages.argmax(), averages


def transform(sample_set: SampleSet, num_components, random_seed=False):
    random_state = None if random_seed else 0
    ica = FastICA(n_components=num_components,
                  algorithm='parallel',
                  whiten='unit-variance',
                  fun='logcosh',
                  max_iter=2000,
                  whiten_solver='eigh',
                  random_state=random_state)
    ica.fit(sample_set.samples)
    return ica, ica.transform(sample_set.samples)
