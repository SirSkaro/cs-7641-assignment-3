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


LETTER_KS_OF_INTEREST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 35, 50, 350, 450, 550, 700, 1000, 1500]
SCRIBE_KS_OF_INTEREST = [2, 3, 4, 100, 500, 1000, 2500, 5000, 7000]
LETTER_MEANS_OF_INTEREST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 35, 50, 100, 130, 140, 150, 200, 500, 1000, 2000]
SCRIBE_MEANS_OF_INTEREST = [2, 3, 4, 5, 100, 300, 500, 550, 600, 750, 1000]


class Combo(Enum):
    LETTER_PCA_KMC = TaskConfig(Task.LETTER_RECOGNITION, pca, kmc, 6, LETTER_KS_OF_INTEREST)
    SCRIBE_PCA_KMC = TaskConfig(Task.SCRIBE_RECOGNITION, pca, kmc, 4, SCRIBE_KS_OF_INTEREST)
    LETTER_PCA_EM = TaskConfig(Task.LETTER_RECOGNITION, pca, em, 6, LETTER_MEANS_OF_INTEREST)
    SCRIBE_PCA_EM = TaskConfig(Task.SCRIBE_RECOGNITION, pca, em, 4, SCRIBE_MEANS_OF_INTEREST)

    LETTER_ICA_KMC = TaskConfig(Task.LETTER_RECOGNITION, ica, kmc, 12, LETTER_KS_OF_INTEREST)
    SCRIBE_ICA_KMC = TaskConfig(Task.SCRIBE_RECOGNITION, ica, kmc, 1, SCRIBE_KS_OF_INTEREST)
    LETTER_ICA_EM = TaskConfig(Task.LETTER_RECOGNITION, ica, em, 12, LETTER_MEANS_OF_INTEREST)
    SCRIBE_ICA_EM = TaskConfig(Task.SCRIBE_RECOGNITION, ica, em, 1, SCRIBE_MEANS_OF_INTEREST)

    LETTER_RP_KMC = TaskConfig(Task.LETTER_RECOGNITION, rp, kmc, 11, LETTER_KS_OF_INTEREST)
    SCRIBE_RP_KMC = TaskConfig(Task.SCRIBE_RECOGNITION, rp, kmc, 6, SCRIBE_KS_OF_INTEREST)
    LETTER_RP_EM = TaskConfig(Task.LETTER_RECOGNITION, rp, em, 11, LETTER_MEANS_OF_INTEREST)
    SCRIBE_RP_EM = TaskConfig(Task.SCRIBE_RECOGNITION, rp, em, 6, SCRIBE_MEANS_OF_INTEREST)

    LETTER_KPCA_KMC = TaskConfig(Task.LETTER_RECOGNITION, k_pca, kmc, 6, LETTER_KS_OF_INTEREST)
    SCRIBE_KPCA_KMC = TaskConfig(Task.SCRIBE_RECOGNITION, k_pca, kmc, 5, SCRIBE_KS_OF_INTEREST)
    LETTER_KPCA_EM = TaskConfig(Task.LETTER_RECOGNITION, k_pca, em, 6, LETTER_MEANS_OF_INTEREST)
    SCRIBE_KPCA_EM = TaskConfig(Task.SCRIBE_RECOGNITION, k_pca, em, 5, SCRIBE_MEANS_OF_INTEREST)


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
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, pca, em)
    model, transformed_samples = pca.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = em.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def ica_km(task: Task):
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, ica, kmc)
    model, transformed_samples = ica.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = kmc.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def ica_em(task: Task):
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, ica, em)
    model, transformed_samples = ica.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = em.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def rp_km(task: Task):
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, rp, kmc)
    model, transformed_samples = rp.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = kmc.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def rp_em(task: Task):
    original_samples = data_utils.get_all_samples(task)
    config = get_config(task, rp, em)
    model, transformed_samples = rp.transform(original_samples, config.target_dimensions)
    transformed_sample_set = SampleSet(transformed_samples, original_samples.labels)

    scores = em.graph_evaluations(transformed_sample_set, config.centroids_of_interest)
    return scores


def kpca_km(task: Task):
    pass


def kpca_em(task: Task):
    pass




