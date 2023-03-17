from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from data_utils import Task, SampleSet
import data_utils

### Docs used:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html

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

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.zaxis.set_ticklabels([])
    plt.show()


def graph_analysis(task: Task):
    sample_set = data_utils.get_all_samples(task)
    num_features = sample_set.samples.shape[1]
    pca, _ = transform(sample_set, num_features)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Explained Variance per Component')
    ax[0].set_xlabel("Components")
    ax[0].set_ylabel("Explained Variance")
    ax[1].set_title('Explained Variance Ratios')

    component_labels = np.arange(1, num_features+1)
    ax[0].plot(component_labels, pca.explained_variance_, marker="o", drawstyle="default", linestyle='solid')
    ax[1].pie(pca.explained_variance_ratio_, labels=component_labels, autopct='%1.1f%%', pctdistance=1.25, labeldistance=.6, radius=1.1)

    plt.show()


def transform(sample_set: SampleSet, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(sample_set.samples)
    return pca, pca.transform(sample_set.samples)
