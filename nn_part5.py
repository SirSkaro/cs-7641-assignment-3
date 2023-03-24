import data_utils
from data_utils import Task, SampleSet
import kmc, em

import tensorflow as tf
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from time import time
from sklearn.base import BaseEstimator

### Docs used:
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/keras/classification
# https://keras.io/api/layers/
# https://keras.io/api/optimizers/


DEFAULT_INITIALIZER = 'glorot_uniform'

class Activation(Enum):
    SIGMOID = ('sig', DEFAULT_INITIALIZER)
    LEAKY_RELU = (tf.keras.layers.LeakyReLU(alpha=0.1), DEFAULT_INITIALIZER)
    PARAMETRIC_LEAKY_RELU = (tf.keras.layers.PReLU(), DEFAULT_INITIALIZER)
    EXPONENTIAL_LINEAR_UNIT = ('elu', DEFAULT_INITIALIZER)
    SCALED_EXPONENTIAL_LINEAR_UNIT = ('selu', 'lecun_normal')


class Optimizer(Enum):
    GRADIENT_DESCENT = lambda: tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5)
    ADA_DELTA = lambda: tf.keras.optimizers.Adadelta(learning_rate=0.9, use_ema=True,
                                             ema_momentum=0.5, ema_overwrite_frequency=100)


def learn(training_set: SampleSet, test_set: SampleSet, hidden_layers: int = 3, units_per_hidden_layer=10,
          activation: Activation = Activation.SCALED_EXPONENTIAL_LINEAR_UNIT,
          optimizer: Optimizer = Optimizer.ADA_DELTA):
    activation_function = activation.value[0]
    initializer = activation.value[1]

    # Build layers
    layers = [tf.keras.layers.Input(shape=(training_set.num_features(),), name='input')]
    for layer_index in range(0, hidden_layers):
        layers.append(tf.keras.layers.Dense(units=units_per_hidden_layer, name=f'hidden{layer_index}',
                                            activation=activation_function, kernel_initializer=initializer))
    layers.append(tf.keras.layers.Dense(units=training_set.num_classes(), name='output'))

    classifier = tf.keras.models.Sequential(layers)
    classifier.compile(
        optimizer=optimizer(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train classifier
    converged = False
    required_improvement = 0.01
    epoch = 0
    epochs_per_run = 500
    previous_best = 0
    while not converged:
        history = classifier.fit(training_set.samples, training_set.labels_as_ints(),
                                 epochs=epoch + epochs_per_run, initial_epoch=epoch,
                                 validation_split=0.2, shuffle=False,
                                 batch_size=128)
        epoch += epochs_per_run
        # epoch_validation_accuracy = np.array(history.history['val_accuracy'][-epochs_per_run:])
        epoch_accuracy = np.array(history.history['accuracy'][-epochs_per_run:])
        current_best = epoch_accuracy.max()
        improvement = current_best - previous_best
        converged = improvement <= required_improvement
        print('improvement for epoch is ~' + str(improvement))
        previous_best = current_best

    test_set.use_label_to_int_map_from(training_set)
    loss, accuracy = classifier.evaluate(test_set.samples, test_set.labels_as_ints(), verbose=2)

    test_error = 1.0 - accuracy
    train_error = 1.0 - history.history['accuracy'][-1]
    validation_error = 1.0 - history.history['val_accuracy'][-1]
    return classifier, test_error, train_error, validation_error


def add_cluster_features(cluster_models, training_set: SampleSet, test_set: SampleSet):
    add_features(training_set, cluster_models)
    add_features(test_set, cluster_models)


def add_features(sample_set: SampleSet, cluster_models):
    features_to_add = [model.predict(sample_set.samples) for model in cluster_models]
    for feature in features_to_add:
        sample_set.samples = np.column_stack((sample_set.samples, feature))


def create_learning_curve():
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Percentage Training Set")
    ax[0].set_ylabel("Error")

    ax[1].set_title("Training Time")
    ax[1].set_xlabel("Percentage Training Set")
    ax[1].set_ylabel("Training Time (in Seconds)")

    task = Task.LETTER_RECOGNITION
    original_data = data_utils.get_all_samples(task)
    cluster_models = [kmc.create_clustering(original_data, 450), em.create_clustering(original_data, 140)]
    percentages = np.linspace(0, 1, 11)[1:-1]
    original_test_errors = []
    original_train_errors = []
    original_validation_errors = []
    original_training_times = []
    augmented_test_errors = []
    augmented_train_errors = []
    augmented_validation_errors = []
    augmented_training_times = []

    for percent_training in percentages:
        # Train network on original data
        training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training, randomize=True)

        original_start = time()
        _, test_error, train_error, validation_error = learn(training_set, test_set, units_per_hidden_layer=10,
                                                            hidden_layers=6, optimizer=Optimizer.ADA_DELTA,
                                                            activation=Activation.SCALED_EXPONENTIAL_LINEAR_UNIT)
        original_end = time()

        original_test_errors.append(test_error)
        original_train_errors.append(train_error)
        original_validation_errors.append(validation_error)
        original_training_times.append(round(original_end - original_start, 2))

        # Train network on augmented data
        add_cluster_features(cluster_models, training_set, test_set)

        augmented_start = time()
        _, test_error, train_error, validation_error = learn(training_set, test_set, units_per_hidden_layer=35,
                                                             hidden_layers=3, optimizer=Optimizer.ADA_DELTA,
                                                             activation=Activation.SCALED_EXPONENTIAL_LINEAR_UNIT)
        augmented_end = time()

        augmented_test_errors.append(test_error)
        augmented_train_errors.append(train_error)
        augmented_validation_errors.append(validation_error)
        augmented_training_times.append(round(augmented_end - augmented_start, 2))

    ax[0].plot(percentages, original_test_errors, label=f'Original Test', marker="o", drawstyle="default", linestyle='dotted')
    ax[0].plot(percentages, original_train_errors, label=f'Original Train', marker="o", drawstyle="default", linestyle='dotted')
    ax[0].plot(percentages, original_validation_errors, label=f'Original Validation', marker="o", drawstyle="default", linestyle='dotted')
    ax[1].plot(percentages, original_training_times, label=f'Original Classifier', marker="o", drawstyle="default", linestyle='dotted')

    ax[0].plot(percentages, augmented_test_errors, label=f'w/ Clusters Test', marker="s", drawstyle="default", linestyle='dashed')
    ax[0].plot(percentages, augmented_train_errors, label=f'w/ Clusters Train', marker="s", drawstyle="default", linestyle='dashed')
    ax[0].plot(percentages, augmented_validation_errors, label=f'w/ Clusters Validation', marker="s", drawstyle="default", linestyle='dashed')
    ax[1].plot(percentages, augmented_training_times, label=f'w/ Cluster Classifier', marker="s", drawstyle="default", linestyle='dashed')

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")

    plt.show()

    return (original_test_errors, original_train_errors, original_validation_errors, original_training_times),\
           (augmented_test_errors, augmented_train_errors, augmented_validation_errors, augmented_training_times)
