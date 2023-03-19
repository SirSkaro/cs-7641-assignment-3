from typing import Tuple

from enum import Enum
import numpy
from sklearn.utils import shuffle


LETTER_FEATURES = ['Box X Position', 'Box Y Position', 'Box Width', 'Box Height', 'Total Pixels',
                 'Pixel Mean X-Coor.', 'Pixel Mean Y-Coor', 'X Variance', 'Y Variance', 'Mean XY Correlation',
                 'Mean of X*X*Y', 'Mean of X*Y*Y', 'X Edge Count Mean', 'Correlation of X-ege with Y', 'Y Edge Count Mean',
                   'Correlation of Y-ege with X']
LETTER_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

SCRIBE_FEATURES = ['Intercolumnar Distance', 'Upper Margin', 'Lower Margin', 'Exploitation', 'Row Number', 'Modular Ratio',
                   'Interlinear Spacing', 'Weight', 'Peak Number', 'Modular Ratio/Interlinear Spacing']
SCRIBE_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y', 'Z']


class TaskMetadata:
    def __init__(self, directory, label_index, data_indexes, data_type, features, classes):
        self.directory = directory
        self.label_index = label_index
        self.data_indexes = data_indexes
        self.data_type = data_type
        self.features = features
        self.classes = classes


class Task(Enum):
    LETTER_RECOGNITION = TaskMetadata('letter recognition', 0, numpy.arange(1, 17, dtype=int), int, LETTER_FEATURES, LETTER_CLASSES)
    SCRIBE_RECOGNITION = TaskMetadata('scribe recognition', 10, numpy.arange(0, 10, dtype=int), float, SCRIBE_FEATURES, SCRIBE_CLASSES)


class SampleSet:
    def __init__(self, samples, labels):
        if not samples.shape[0] == labels.shape[0]:
            raise ValueError('mismatch in number of labels vs samples')

        self.samples = samples
        self.labels = labels
        self._label_int_map = None

    def size(self) -> int:
        return self.labels.size

    def num_features(self) -> int:
        return self.samples.shape[1]

    def num_classes(self) -> int:
        return len(numpy.unique(self.labels))

    def labels_as_ints(self):
        label_map = self._label_int_map or self._create_labels_to_int_map()
        map_function = numpy.vectorize(lambda label: label_map[label])
        return map_function(self.labels)

    def use_label_to_int_map_from(self, other: 'SampleSet'):
        self._label_int_map = other._label_int_map

    def _create_labels_to_int_map(self):
        self._label_int_map = {}
        for index, label in enumerate(numpy.unique(self.labels)):
            self._label_int_map[label] = index
        return self._label_int_map


def parse_data(task: Task) -> Tuple[numpy.ndarray, numpy.array]:
    task_metadata = task.value
    filename = './datasets/' + task_metadata.directory + '/data'
    dataset = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=task_metadata.data_indexes,
        dtype=task_metadata.data_type
    )
    labels = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=task_metadata.label_index,
        dtype=str
    )

    return dataset, labels


def partition_samples(samples: numpy.ndarray, labels: numpy.array, percent_training: float = 0.9, randomize: bool = False):
    sample_count = labels.size
    training_set_size = int(sample_count * percent_training)

    if randomize:
        samples, labels = shuffle(samples, labels)

    training_set = SampleSet(samples[:training_set_size], labels[:training_set_size])
    test_set = SampleSet(samples[training_set_size:], labels[training_set_size:])
    return training_set, test_set


def get_training_and_test_sets(task: Task, percent_training: float = 0.9, randomize: bool = False) -> Tuple[SampleSet, SampleSet]:
    samples, labels = parse_data(task)
    return partition_samples(samples, labels, percent_training, randomize)


def get_training_validation_and_test_sets(task: Task,
                                          percent_training: float = 0.9,
                                          percent_validation: float = 0.2,
                                          randomize: bool = False) -> Tuple[SampleSet, SampleSet, SampleSet]:
    training_set, test_set = get_training_and_test_sets(task, percent_training=percent_training, randomize=randomize)
    training_set, validation_set = partition_samples(training_set.samples, training_set.labels, 1 - percent_validation)
    return training_set, validation_set, test_set


def get_all_samples(task: Task):
    samples, labels = parse_data(task)
    return SampleSet(samples, labels)