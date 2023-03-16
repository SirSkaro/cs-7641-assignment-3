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


def create_graph(task: Task):
    components_to_try = np.arange(2, 10002, 1000)
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

    best = sorted(score_cluster_tuples, key=lambda tup: tup[2], reverse=True)[0]
    best_k = best[0]
    clustering = best[1]
    score = best[2]

    return (best_k, clustering, score), all_scores


def create_clustering(sample_set: SampleSet, k: int, init: MeanInit, trials: int):
    clusters = GaussianMixture(n_components=k,
                               random_state=0,
                               covariance_type='full',
                               tol=1e-3,
                               n_init=trials,
                               max_iter=100,
                               init_params=init.value)
    clusters.fit(sample_set.samples)

    return clusters

