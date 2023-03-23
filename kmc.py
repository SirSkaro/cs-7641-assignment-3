from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from data_utils import Task, SampleSet
import data_utils

from enum import Enum

### Docs used:
# https://scikit-learn.org/stable/modules/clustering.html#clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py


class MeanInit(Enum):
    RANDOM = 'random'
    KM_PP = 'k-means++'


def graph_evaluations(sample_set: SampleSet, ks_of_interest):
    silhouette_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []

    for k in ks_of_interest:
        print(f'Evaluating clustering for {k}...')
        scores = evaluate_clustering(sample_set, k)
        silhouette_scores.append(scores[0])
        homogeneity_scores.append(scores[1])
        completeness_scores.append(scores[2])
        v_measure_scores.append(scores[3])

    fig, ax = plt.subplots(1, 1)
    ax.plot(ks_of_interest, silhouette_scores, label='Silhouette Scores', marker="o", drawstyle="default", linestyle='solid')
    ax.plot(ks_of_interest, homogeneity_scores, label='Homogeneity Scores', marker="o", drawstyle="default", linestyle='solid')
    ax.plot(ks_of_interest, completeness_scores, label='Completeness Scores', marker="o", drawstyle="default", linestyle='solid')
    ax.plot(ks_of_interest, v_measure_scores, label='V-Measure Scores', marker="o", drawstyle="default", linestyle='solid')

    ax.legend(loc="best")
    plt.show()
    return silhouette_scores, homogeneity_scores, completeness_scores, v_measure_scores


def evaluate_clustering(sample_set: SampleSet, k: int):
    clustering = create_clustering(sample_set, k, MeanInit.KM_PP)
    predicted_labels = clustering.predict(sample_set.samples)

    average_silhouette_score = metrics.silhouette_score(sample_set.samples, clustering.labels_)
    homogeneity = metrics.homogeneity_score(sample_set.labels, predicted_labels)
    completeness = metrics.completeness_score(sample_set.labels, predicted_labels)
    v_measure = metrics.v_measure_score(sample_set.labels, predicted_labels)

    return average_silhouette_score, homogeneity, completeness, v_measure


def create_graph(sample_set: SampleSet, ks_to_try: np.array):
    (_, clustering, best_average_score), all_average_scores = find_k(sample_set, MeanInit.KM_PP, trials_per_k=1, ks_to_try=ks_to_try)
    graph_scores(sample_set, clustering, best_average_score, all_average_scores, ks_to_try)


# Logic largely borrowed from docs listed above
def graph_scores(sample_set: SampleSet, clustering: KMeans, best_average_score, average_scores_for_all_ks, ks_to_try: np.array):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim([-1, 1])   # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_ylim([0, len(sample_set.samples) + (clustering.n_clusters + 1) * 10])  # insert blank space between silhouette

    # get silhouette metrics
    cluster_labels = clustering.predict(sample_set.samples)
    sample_silhouette_values = metrics.silhouette_samples(sample_set.samples, cluster_labels)

    # construct silhouette analysis graph
    ax1.set_title("Silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster")
    y_lower = 10
    for k in ks_to_try:
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == k]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(k) / clustering.n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))  # Label the silhouette plots with their cluster numbers at the middle
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.axvline(x=best_average_score, color="red", linestyle="--")  # add average line
    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    # construct average silhouette score per k
    ax2.set_title("Average silhouette coefficient value")
    ax2.set_xlabel("k (# of clusters)")
    ax2.set_ylabel("Silhouette coefficient values")
    clusters = np.arange(2, len(average_scores_for_all_ks) + 2)
    ax2.plot(clusters, average_scores_for_all_ks, marker="o", drawstyle="default", linestyle='solid')

    plt.show()


def find_k(sample_set: SampleSet, init: MeanInit, trials_per_k: int, ks_to_try: np.array):
    score_cluster_tuples = []
    all_average_scores = []
    for k in ks_to_try:
        print(f'Creating cluster for {k}...')
        clustering = create_clustering(sample_set, k, init, trials_per_k)
        average_silhouette_score = metrics.silhouette_score(sample_set.samples, clustering.labels_)
        score_cluster_tuples.append((k, clustering, average_silhouette_score))
        all_average_scores.append(average_silhouette_score)
        print(f'\tscore: {average_silhouette_score}')

    best = sorted(score_cluster_tuples, key=lambda tup: tup[2], reverse=True)[0]
    best_k = best[0]
    clustering = best[1]
    avg_score = best[2]

    return (best_k, clustering, avg_score), all_average_scores


def create_clustering(sample_set: SampleSet, k: int, init: MeanInit = MeanInit.KM_PP, trials: int = 1):
    clusters = KMeans(n_clusters=k,
                      init=init.value,
                      n_init=trials,
                      algorithm='lloyd',
                      random_state=None)
    clusters.fit(sample_set.samples)

    return clusters

