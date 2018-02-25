# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from keras_models import dense_model
from data_management import read_all_files
from data_management import dataset_location
from data_management import prepare_data

from services import KerasGridSearchService
# Function to create model, required for KerasClassifier

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

TEST_PHASE = True


def gs_find():
    """
    Grid search to find the best results.

    :return:
    """

    # load dataset
    train_data, train_class, validation_data, validation_class, test_data, test_class = \
        read_all_files(dataset_location + 'var_1_range_1_class_1x2/')

    x_train, y_train_encoded, x_validation, y_validation_encoded, x_test, y_test_encoded = \
        prepare_data(train_data, train_class, validation_data, validation_class, test_data, test_class)

    # Use this when testing
    if TEST_PHASE:
        x_train = np.concatenate((x_train, x_validation), axis=0)
        x_validation = x_test

        y_train_encoded = np.concatenate((y_train_encoded, y_validation_encoded), axis=0)
        y_validation_encoded = y_test_encoded

    print("Start search")
    # create model
    model = KerasClassifier(build_fn=dense_model, verbose=0)
    # define the grid search parameters

    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    batch_size = [100]
    epochs = [50, 100]

    weight_constraint = [2, 3, 4]
    neurons = [20, 51]
    dropout_rate = [0.2, 0.5, 0.7]

    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      # optimizer=optimizer,
                      # init_mode=init_mode,
                      # activation=activation,
                      dropout_rate=dropout_rate,
                      weight_constraint=weight_constraint,
                      neurons=neurons
                      )

    avg_scores = {}
    # while len(avg_scores) > 1:
    tries = 3
    for i in range(0, tries):
        grid_results = KerasGridSearchService.keras_grid_search(dense_model, param_grid,
                                                                x_train, y_train_encoded,
                                                                x_validation, y_validation_encoded,
                                                                keras_backend=K)
        for gr in grid_results:
            avg_scores.setdefault(gr, 0)
            avg_scores[gr] += grid_results[gr]

    for avs in avg_scores:
        avg_scores[avs] /= tries
    print('')
    print('')

    print('Scores: ' + str(avg_scores))
    #  In a while loop should try until you get one best score
    #  make arrays from avg_scores --> param_grid
    #  --- get the value that has mos occurrences drop all its members from the array
    #  --- and get all the other params that it has good values with

