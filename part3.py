from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from data_utils import Task, SampleSet
import data_utils
import em
import ica
import kmc
import k_pca
import pca
import rp

from enum import Enum


class TaskConfig:
    def __init__(self, task, reduction, clustering, target_dimensions, centroids_of_interest):
        self.task = task
        self.reduction = reduction
        self.clustering = clustering
        self.target_dimensions = target_dimensions
        self.centroids_of_interest = centroids_of_interest


LETTER_Ks_OF_INTEREST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 35, 50, 350, 450, 550, 700, 1000, 1500]
SCRIBE_Ks_OF_INTEREST = [2, 3, 4, 100, 500, 1000, 2500, 5000, 7000]


class Combo(Enum):
    LETTER_PCA_KM = TaskConfig(Task.LETTER_RECOGNITION, pca, kmc, 6, LETTER_Ks_OF_INTEREST)
    SCRIBE_PCA_KM = TaskConfig(Task.SCRIBE_RECOGNITION, pca, kmc, 4, SCRIBE_Ks_OF_INTEREST)


def get_config(task: Task, reduction, clustering) -> TaskConfig:
    for config in [combo.value for combo in Combo]:
        if config.task == task and config.reduction == reduction and config.clustering == clustering:
            return config
    raise ValueError('Config does not exist')


def pca_km(task: Task):
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, pca, kmc)
    model, transformed_samples = pca.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = kmc.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def pca_em(task: Task):
    pass


def ica_km(task: Task):
    pass


def ica_em(task: Task):
    pass


def rp_km(task: Task):
    pass


def rp_em(task: Task):
    pass


def kpca_km(task: Task):
    pass


def kpca_em(task: Task):
    pass




