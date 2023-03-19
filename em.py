from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from data_utils import Task, SampleSet
import data_utils

from enum import Enum


### Docs used:
# https://scikit-learn.org/stable/modules/clustering.html#clustering
# https://scikit-learn.org/stable/modules/mixture.html
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

class MeanInit(Enum):
    RANDOM = 'random'
    KM_PP = 'k-means++'


def graph_evaluations(sample_set: SampleSet, components_of_interest):
    bic_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []

    for components in components_of_interest:
        print(f'Evaluating {components} components...')
        scores = evaluate_clustering(sample_set, components)
        bic_scores.append(scores[0])
        homogeneity_scores.append(scores[1])
        completeness_scores.append(scores[2])
        v_measure_scores.append(scores[3])

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Evaluation Scores')
    ax[0].set_xlabel("Components")
    ax[0].set_ylabel("Score")
    ax[1].set_title('BIC Scores')
    ax[1].set_xlabel("Components")
    ax[1].set_ylabel("Score")

    ax[0].plot(components_of_interest, homogeneity_scores, label='Homogeneity Scores', marker="o", drawstyle="default", linestyle='solid')
    ax[0].plot(components_of_interest, completeness_scores, label='Completeness Scores', marker="o", drawstyle="default", linestyle='solid')
    ax[0].plot(components_of_interest, v_measure_scores, label='V-Measure Scores', marker="o", drawstyle="default", linestyle='solid')

    ax[1].plot(components_of_interest, bic_scores, marker="o", drawstyle="default", linestyle='solid')

    ax[0].legend(loc="best")
    plt.show()
    return bic_scores, homogeneity_scores, completeness_scores, v_measure_scores


def evaluate_clustering(sample_set: SampleSet, component_count: int):
    clustering = create_clustering(sample_set, component_count, MeanInit.KM_PP)
    predicted_labels = clustering.predict(sample_set.samples)

    bic = clustering.bic(sample_set.samples)
    homogeneity = metrics.homogeneity_score(sample_set.labels, predicted_labels)
    completeness = metrics.completeness_score(sample_set.labels, predicted_labels)
    v_measure = metrics.v_measure_score(sample_set.labels, predicted_labels)

    return bic, homogeneity, completeness, v_measure


def create_graph(task: Task):
    components_to_try = np.arange(250, 751, 100)
    sample_set = data_utils.get_all_samples(task)
    (best_component, clustering, score), all_scores = find_best_cluster(sample_set, MeanInit.KM_PP,
                                                                        trials_per_k=1,
                                                                        components_to_try=components_to_try)
    graph_scores(all_scores, components_to_try, best_component)


def graph_scores(all_scores, component_counts, best_component):
    fig, ax = plt.subplots(1)
    ax.set_title("BIC for the various clusters")
    ax.set_xlabel("# of components")
    ax.set_ylabel("BIC")

    ax.plot(component_counts, all_scores, marker="o", drawstyle="default", linestyle='solid')
    ax.axvline(x=best_component, color="red", linestyle="--")
    plt.show()


def find_best_cluster(sample_set: SampleSet, init: MeanInit, trials_per_k: int, components_to_try: np.array):
    score_cluster_tuples = []
    all_scores = []
    for k in components_to_try:
        print(f'Creating cluster for {k}...')
        clustering = create_clustering(sample_set, k, init, trials_per_k)
        bic = clustering.bic(sample_set.samples)
        score_cluster_tuples.append((k, clustering, bic))
        all_scores.append(bic)
        print(f'\tscore: {bic}')

    best = sorted(score_cluster_tuples, key=lambda tup: tup[2])[0]
    best_k = best[0]
    clustering = best[1]
    score = best[2]

    return (best_k, clustering, score), all_scores


def create_clustering(sample_set: SampleSet, k: int, init: MeanInit, trials: int = 1):
    clusters = GaussianMixture(n_components=k,
                               random_state=0,
                               covariance_type='full',
                               tol=1e-3,
                               n_init=trials,
                               max_iter=50,
                               init_params=init.value)
    clusters.fit(sample_set.samples)

    return clusters

