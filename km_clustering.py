from sklearn.cluster import KMeans
import numpy as np
from data_utils import Task
import data_utils
from enum import Enum

### Docs used:
# https://scikit-learn.org/stable/modules/clustering.html#clustering

class MEAN_INITIALIZATION(Enum):
    RANDOM = 'random'
    KM_PP = 'k-means++'

def cluster(task: Task, k: int, init: MEAN_INITIALIZATION, trials: int):
    samples = data_utils.get_all_samples(task)

    clusters = KMeans(n_clusters=k,
                      init=init.value,
                      n_init=trials,
                      verbose=1,
                      algorithm='lloyd',
                      random_state=None
                      ).fit(samples)

    return clusters

