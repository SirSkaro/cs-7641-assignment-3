import matplotlib.ticker
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
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py


class MeanInit(Enum):
    RANDOM = 'random'
    KM_PP = 'k-means++'


class CandidateCluster:
    def __init__(self, cluster_class, sample_count):
        self.cluster_class = cluster_class
        self.sample_count = sample_count


class ClusterAnalysis:
    def __init__(self, label, candidate_clusters, total):
        self.label = label
        self.candidate_clusters = sorted(candidate_clusters, key=lambda candidate: candidate.sample_count, reverse=True)
        self.total = total

    def __str__(self):
        most_likely_cluster = self.most_likely_cluster()
        sample_count = most_likely_cluster.sample_count
        confidence = round(sample_count/self.total * 100, 4)
        return f'{self.label} -> {most_likely_cluster.cluster_class} ({sample_count}/{self.total} | {confidence}%)'

    def most_likely_cluster(self) -> CandidateCluster:
        return self.candidate_clusters[0]


def create_graph(task: Task):
    sample_set = data_utils.get_all_samples(task)
    (_, clustering, _), all_average_scores = find_k(sample_set, MeanInit.KM_PP, trials_per_k=5)
    graph_scores(task, sample_set, clustering, all_average_scores)


# Logic largely borrowed from docs listed above
def graph_scores(task: Task, sample_set: SampleSet, clustering: KMeans, average_scores_for_all_ks):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim([-1, 1])   # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_ylim([0, len(sample_set.samples) + (clustering.n_clusters + 1) * 10])  # insert blank space between silhouette

    # get silhouette metrics
    cluster_labels = clustering.predict(sample_set.samples)
    silhouette_avg = average_scores_for_all_ks[clustering.n_clusters - 2]
    sample_silhouette_values = metrics.silhouette_samples(sample_set.samples, cluster_labels)

    # construct silhouette analysis graph
    ax1.set_title("Silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster")
    y_lower = 10
    for i in range(clustering.n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / clustering.n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Label the silhouette plots with their cluster numbers at the middle
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")  # add average line
    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    # construct average silhouette score per k
    ax2.set_title("Average silhouette coefficient value")
    ax2.set_xlabel("k (# of clusters)")
    ax2.set_ylabel("Silhouette coefficient values")
    clusters = np.arange(2, len(average_scores_for_all_ks) + 2)
    ax2.plot(clusters, average_scores_for_all_ks, marker="o", drawstyle="default", linestyle='solid')

    plt.show()


def print_scores_to_csv(filename: str, scores: np.ndarray):
    np.savetxt('outputs/'+filename, scores, delimiter=',', fmt="%1.6f")


def find_k(sample_set: SampleSet, init: MeanInit, trials_per_k: int):
    score_cluster_tuples = []
    all_average_scores = []
    for k in range(2, 201):
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


def create_clustering(sample_set: SampleSet, k: int, init: MeanInit, trials: int):
    clusters = KMeans(n_clusters=k,
                      init=init.value,
                      n_init=trials,
                      algorithm='lloyd',
                      random_state=None)
    clusters.fit(sample_set.samples)

    return clusters #analyze(clusters, sample_set)


def analyze(clusters: KMeans, sample_set: SampleSet):
    classes = np.unique(sample_set.labels)
    class_cluster_map = {}
    for clazz in classes:
        class_mask = np.where(sample_set.labels == clazz)
        samples_of_class = sample_set.samples[class_mask]
        predicted_clusters = clusters.predict(samples_of_class)
        occurrences_of_clusters = np.bincount(predicted_clusters)
        candidate_clusters = [CandidateCluster(tup[0], tup[1]) for tup in enumerate(occurrences_of_clusters)]
        analysis = ClusterAnalysis(label=clazz, candidate_clusters=candidate_clusters, total=occurrences_of_clusters.sum())
        class_cluster_map[clazz] = analysis

    return class_cluster_map

