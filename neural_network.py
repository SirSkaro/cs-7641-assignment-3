import data_utils
from data_utils import Task

import tensorflow as tf
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from time import time

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


def learn(task: Task, shuffle: bool = False, hidden_layers: int = 3, percent_training: float = 0.9,
          activation: Activation = Activation.SCALED_EXPONENTIAL_LINEAR_UNIT,
          optimizer: Optimizer = Optimizer.GRADIENT_DESCENT
          ):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training=percent_training, randomize=shuffle)
    activation_function = activation.value[0]
    initializer = activation.value[1]

    # Build layers
    layers = [tf.keras.layers.Input(shape=(training_set.num_features(),), name='input')]
    for layer_index in range(0, hidden_layers):
        layers.append(tf.keras.layers.Dense(units=10, name=f'hidden{layer_index}', activation=activation_function, kernel_initializer=initializer))
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


def create_learning_curve():
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Percentage Training Set")
    ax[0].set_ylabel("Error")

    ax[1].set_title("Training Time")
    ax[1].set_xlabel("Percentage Training Set")
    ax[1].set_ylabel("Training Time (in Seconds)")

    percentages = np.linspace(0, 1, 11)[1:-1]
    for task, name, linestyle, hidden_layers in [(Task.SCRIBE_RECOGNITION, 'Scribe', 'dotted', 6),
                                         (Task.LETTER_RECOGNITION, 'Letter', 'dashed', 6)]:
        test_errors = []
        train_errors = []
        validation_errors = []
        training_times = []
        for percent_training in percentages:
            start = time()
            _, test_error, train_error, validation_error = learn(task, percent_training=percent_training, shuffle=True,
                                               hidden_layers=hidden_layers, optimizer=Optimizer.ADA_DELTA,
                                               activation=Activation.SCALED_EXPONENTIAL_LINEAR_UNIT)
            end = time()

            test_errors.append(test_error)
            train_errors.append(train_error)
            validation_errors.append(validation_error)
            training_times.append(round(end - start, 2))

        ax[0].plot(percentages, test_errors, label=f'{name} Test Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[0].plot(percentages, train_errors, label=f'{name} Train Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[0].plot(percentages, validation_errors, label=f'{name} Validation Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[1].plot(percentages, training_times, label=f'{name} Classifier', marker="o", drawstyle="steps-post", linestyle=linestyle)

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
