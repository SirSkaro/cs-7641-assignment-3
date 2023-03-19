from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import matplotlib.pyplot as plt
import pca

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
    pca_errors = []
    mean_errors = []
    best_errors = []
    worst_errors = []
    components_to_try = np.arange(1, num_features + 1)

    def calc_sum_of_squared_error(model, transformed_data):
        reconstruction = model.inverse_transform(transformed_data)
        return ((sample_set.samples - reconstruction) ** 2).sum()

    for num_components in components_to_try:
        errors = []
        best_error = float("inf")
        best_model = None

        # calculate PCA reconstruction error as metric
        pca_model, transformed_pca_data = pca.transform(sample_set, num_components)
        pca_error = calc_sum_of_squared_error(pca_model, transformed_pca_data)
        pca_errors.append(pca_error)

        # create many RP models and calculation reconstruction error
        for trial in range(trials):
            rp_model, transformed_rp_data = transform(sample_set, num_components)
            rp_error = calc_sum_of_squared_error(rp_model, transformed_rp_data)
            # keep best
            if rp_error < best_error:
                best_error = rp_error
                best_model = rp_model
            errors.append(rp_error)
        best_error_model_pairs.append((best_model, best_error))
        errors = np.array(errors)
        mean_errors.append(errors.mean())
        best_errors.append(best_error)
        worst_errors.append(errors.max())

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Reconstruction error over {trials} trials')
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Sum of squared error")

    ax.bar(components_to_try, mean_errors, width=0.5, label='Mean error')
    ax.bar(components_to_try, best_errors, alpha=0.75, width=0.25, label='Best error')
    ax.bar(components_to_try, worst_errors, alpha=0.5, width=0.75, label='Worst error')
    ax.bar(components_to_try, pca_errors, alpha=0.3, width=1.2, label='PCA error')
    plt.legend(loc='best')
    plt.show()

    return best_error_model_pairs, np.array(mean_errors), np.array(pca_errors)


def transform(sample_set: SampleSet, num_components):
    rca = GaussianRandomProjection(n_components=num_components)
    rca.fit(sample_set.samples)
    return rca, rca.transform(sample_set.samples)
