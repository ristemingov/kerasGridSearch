from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm


def dense_model(input_dim=103, output_dim=3,
                optimizer='adam',
                activation='relu',
                dropout_rate=0.0,
                weight_constraint=0,
                init_mode='uniform',
                neurons=50,
                **_):
    """
    Creates a keras fully connected neural network.
    :param input_dim:
    :param output_dim:
    :param optimizer:
    :param activation:
    :param dropout_rate:
    :param weight_constraint:
    :param init_mode:
    :param neurons:
    :param _:
    :return:
    """
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, kernel_initializer=init_mode, activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
