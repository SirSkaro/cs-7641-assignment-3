from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from data_utils import Task, SampleSet
import data_utils

### Docs used:
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA


def plot_3d(task: Task, kernels, percent_training: float = 0.95):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training)
    fig = plt.figure(figsize=plt.figaspect(0.5))

    def plot_kernel(kernel, percent_training, ax1, ax2, ax3):
        kpca, transformed_training_data = transform(training_set, kernel, 3)
        transformed_test_data = kpca.transform(test_set.samples)

        ax1.scatter(
            transformed_training_data[:, 0],
            transformed_training_data[:, 1],
            transformed_training_data[:, 2],
            c=training_set.labels.astype('|S1').view(np.uint8),
            cmap=plt.cm.Set1,
            edgecolor="k",
            s=40,
        )
        ax1.set_title(f'Transformed training set ({percent_training * 100}%) using {kernel.upper()}')
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
        ax2.set_title(f'Transformed test set ({(round(1.0-percent_training, 2)) * 100}%) using {kernel.upper()}')
        ax2.set_xlabel("1st eigenvector")
        ax2.xaxis.set_ticklabels([])
        ax2.set_ylabel("2nd eigenvector")
        ax2.yaxis.set_ticklabels([])
        ax2.set_zlabel("3rd eigenvector")
        ax2.zaxis.set_ticklabels([])

        ax3.bar([1, 2, 3], kpca.eigenvalues_)
        ax3.set_title("Eigenvalues")
        ax3.set_xlabel("Component")
        ax3.set_ylabel("Eigenvalue")

    num_rows = len(kernels)
    spec = fig.add_gridspec(ncols=3, nrows=num_rows)

    for index, kernel in enumerate(kernels):
        ax1 = fig.add_subplot(spec[index, 0], projection='3d')
        ax2 = fig.add_subplot(spec[index, 1], projection='3d')
        ax3 = fig.add_subplot(spec[index, 2])
        print('Creating plots for ' + kernel)
        plot_kernel(kernel, percent_training, ax1, ax2, ax3)

    plt.show()


def graph_analysis(task: Task, kernels, percent_training: float = 0.95):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training)
    num_features = training_set.samples.shape[1]
    num_kernels = len(kernels)

    fig, ax = plt.subplots(num_kernels, 2)
    for index, kernel in enumerate(kernels):
        kpca, _ = transform(training_set, kernel, num_features)
        eigenvalue_ratio = kpca.eigenvalues_ / kpca.eigenvalues_.max()

        ax[index][0].set_title(f'Eigenvalues per component ({kernel.upper()})')
        ax[index][0].xaxis.set_ticklabels([])
        ax[index][0].set_ylabel("Eigenvalues")

        component_labels = np.arange(1, num_features+1)
        ax[index][0].plot(component_labels, kpca.eigenvalues_, marker="o", drawstyle="default", linestyle='solid')
        ax[index][1].pie(eigenvalue_ratio, labels=component_labels, autopct='%1.1f%%', pctdistance=1.25, labeldistance=.6, radius=1.1)

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
