from __future__ import print_function
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe

import tensorflow as tf
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def ready_train(input_shape=(128, 128, 3), classes=2):
    def set_model(build_model):
        model = build_model(input_shape, classes)

        def trainer(x_train, y_train, x_test, y_test):
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.rmsprop(
                              lr=0.0001, decay=1e-6),
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=32,  # {{choice([32, 64, 128])}},
                      epochs=100,
                      verbose=2,
                      validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test, verbose=0)
            print('Test accuracy:', acc)
            return {'loss': -acc, 'status': STATUS_OK, 'model': model}
        return trainer
    return set_model


def simple_cnn(x_train, y_train, x_test, y_test):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same',
                            input_shape=input_shape))
    model.add(Activation('selu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('selu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('selu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,  # {{choice([32, 64, 128])}},
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def cnn(x_train, y_train, x_test, y_test):
    classes = y_test.shape[-1]

    model = Sequential()

    model.add(Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7)])}},
        padding='same',
        activation='selu',
        input_shape=(128, 128, 3)))
    model.add(Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5)])}},
        activation='selu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5)])}},
        padding='same',
        activation='selu',
    ))
    model.add(Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5)])}},
        activation='selu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('selu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,  # {{choice([32, 64, 128])}},
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def squeezenet(x_train, y_train, x_test, y_test):
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras import backend as K
    from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
    from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.models import Model
    from keras.engine.topology import get_source_inputs
    from keras.utils import get_file
    from keras.utils import layer_utils
    #from keras_squeezenet.squeezenet import fire_module

    def fire_module(x, squeeze=16, expand=64):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        with tf.variable_scope('fire_module'):
            x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
            x = Activation('selu')(x)

            left = Convolution2D(expand, (1, 1), padding='valid')(x)
            left = Activation('selu')(left)

            right = Convolution2D(expand, (3, 3), padding='same')(x)
            right = Activation('selu')(right)

            x = concatenate([left, right], axis=channel_axis)
        return x

    include_top = True
    weights = None
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    if weights not in {'imagenet', None}:
        raise ValueError('The  argument should be either '
                         ' (random initialization) or  '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using  as imagenet with '
                         ' as true,  should be 1000')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])}},
        strides=(2, 2),
        padding='valid',
        activation='selu',
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([16, 32])}}
    expand = {{choice([48, 64])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([16, 32, 64])}}
    expand = {{choice([64, 128, 256])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([24, 48, 96])}}
    expand = {{choice([96, 192, 384])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    squeeze = {{choice([32, 64, 128])}}
    expand = {{choice([128, 256, 512])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation='selu',
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,  # {{choice([32, 64, 128])}},
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def mysqueeze(x_train, y_train, x_test, y_test):
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras import backend as K
    from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
    from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.models import Model
    from keras.engine.topology import get_source_inputs
    from keras.utils import get_file
    from keras.utils import layer_utils

    def fire_module(x, squeeze=16, expand=64):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        with tf.variable_scope('fire_module'):
            x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
            x = Activation('selu')(x)

            left = Convolution2D(expand, (1, 1), padding='valid')(x)
            left = Activation('selu')(left)

            right = Convolution2D(expand, (3, 3), padding='same')(x)
            right = Activation('selu')(right)

            x = concatenate([left, right], axis=channel_axis)
        return x

    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])}},
        strides=(2, 2),
        padding='valid',
        activation='selu',
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    nth = 1
    n_layer = conditional({{choice([1, 2, 3, 4])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        nth += 1

    n_layer = conditional({{choice([1, 2, 3])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)

        nth += 1

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation='selu',
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
