from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from data_utils import Task, SampleSet
import data_utils

### Docs used:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html


def plot_3d(task: Task, kernel: str, percent_training: float = 0.75):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training)
    kpca, transformed_training_data = transform(training_set, kernel, 3)
    transformed_test_data = kpca.transform(test_set.samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), subplot_kw=dict(projection='3d'))
    ax1.scatter(
        transformed_training_data[:, 0],
        transformed_training_data[:, 1],
        transformed_training_data[:, 2],
        c=training_set.labels.astype('|S1').view(np.uint8),
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )
    ax1.set_title("First three PCA directions for training set")
    ax1.set_xlabel("1st eigenvector")
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel("2nd eigenvector")
    ax1.yaxis.set_ticklabels([])
    ax1.set_zlabel("3rd eigenvector")
    ax1.zaxis.set_ticklabels([])

    ax2.scatter(
        transformed_test_data[:, 0],
        transformed_test_data[:, 1],
        transformed_test_data[:, 2],
        c=test_set.labels.astype('|S1').view(np.uint8),
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )
    ax2.set_title("First three PCA directions for test set")
    ax2.set_xlabel("1st eigenvector")
    ax2.xaxis.set_ticklabels([])
    ax2.set_ylabel("2nd eigenvector")
    ax2.yaxis.set_ticklabels([])
    ax2.set_zlabel("3rd eigenvector")
    ax2.zaxis.set_ticklabels([])

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


def transform(sample_set: SampleSet, kernel: str, n_components: int = None):
    kpca = KernelPCA(n_components=n_components,
                     kernel=kernel,
                     degree=10,
                     eigen_solver='randomized',
                     random_state=0,
                     tol=0,
                     n_jobs=4)
    kpca.fit(sample_set.samples)
    return kpca, kpca.transform(sample_set.samples)
