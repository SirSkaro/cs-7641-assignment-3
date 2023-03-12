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


def graph_scores(filename: str):
    pass


def print_scores_to_csv(filename: str, scores: np.ndarray):
    np.savetxt('outputs/'+filename, scores, delimiter=',', fmt="%1.6f")


def find_k(task: Task, init: MeanInit, trials_per_k: int):
    sample_set = data_utils.get_all_samples(task)
    score_cluster_tuples = []
    for k in range(2, 4):
        print(f'Creating cluster for {k}...')
        clustering = create_clustering(sample_set, k, init, trials_per_k)
        average_silhouette_score = metrics.silhouette_score(sample_set.samples, clustering.labels_)
        score_cluster_tuples.append((k, clustering, average_silhouette_score))
        print(f'\tscore: {average_silhouette_score}')

    best = sorted(score_cluster_tuples, key=lambda tup: tup[2], reverse=True)[0]
    best_k = best[0]
    clustering = best[1]
    avg_score = best[2]

    print(f'Best k is {best_k}. Calculating individual scores...')
    individual_scores = metrics.silhouette_samples(sample_set.samples, clustering.labels_)
    return best_k, clustering, avg_score, individual_scores


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

