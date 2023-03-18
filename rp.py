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


def graph_analysis(task: Task, trials: int):
    sample_set = data_utils.get_all_samples(task)
    num_features = sample_set.samples.shape[1]
    best_error_model_pairs = []
    mean_errors = []
    best_errors = []
    worst_errors = []

    components_to_try = np.arange(1, num_features + 1)

    for num_components in components_to_try:
        errors = []
        best_error = float("inf")
        best_model = None
        worst_error = float("-inf")
        for trial in range(trials):
            rp, transformed_data = transform(sample_set, num_components)
            # compute reconstruction error
            reconstruction = rp.inverse_transform(transformed_data)
            sum_of_squared_error = ((sample_set.samples - reconstruction)**2).sum()
            # keep best
            if sum_of_squared_error < best_error:
                best_error = sum_of_squared_error
                best_model = rp
            errors.append(sum_of_squared_error)
        best_error_model_pairs.append((best_model, best_error))
        errors = np.array(errors)
        mean_errors.append(errors.mean())
        best_errors.append(best_error)
        worst_errors.append(errors.max())

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Reconstruction error')
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Sum of squared error")

    ax.bar(components_to_try, mean_errors, label='Mean error')
    ax.bar(components_to_try, best_errors, alpha=0.5, width=0.75, label='Best error')
    ax.bar(components_to_try, worst_errors, alpha=0.1, width=0.5, label='Worst error')
    plt.legend(loc='best')
    plt.show()

    return best_error_model_pairs


def transform(sample_set: SampleSet, num_components):
    rca = GaussianRandomProjection(n_components=num_components)
    rca.fit(sample_set.samples)
    return rca, rca.transform(sample_set.samples)
