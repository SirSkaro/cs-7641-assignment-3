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
        self.cluster_class = cluster_class,
        self.sample_count = sample_count


class ClusterAnalysis:
    def __init__(self, label, cluster_class, correct, total):
        self.label = label
        self.cluster = cluster_class
        self.correct = correct
        self.total = total

    def __str__(self):
        confidence = round(self.correct/self.total * 100, 4)
        return f'{self.label} -> {self.cluster} ({self.correct}/{self.total} | {confidence}%)'


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
        likely_cluster_class = np.argmax(occurrences_of_clusters)
        analysis = ClusterAnalysis(label=clazz, cluster_class=likely_cluster_class,
                                   correct=occurrences_of_clusters[likely_cluster_class],
                                   total=occurrences_of_clusters.sum())
        class_cluster_map[clazz] = analysis

        if likely_cluster_class in class_cluster_map:
            print('Two classes mapped to the same cluster!')

    return class_cluster_map
