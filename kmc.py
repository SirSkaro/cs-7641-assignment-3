from sklearn.cluster import KMeans
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


def cluster(task: Task, k: int, init: MeanInit, trials: int):
    sample_set = data_utils.get_all_samples(task)

    clusters = KMeans(n_clusters=k,
                      init=init.value,
                      n_init=trials,
                      algorithm='lloyd',
                      random_state=None)
    clusters.feature_names_in_ = task.value.features
    clusters.fit(sample_set.samples)

    return clusters, analyze(clusters, sample_set)


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
