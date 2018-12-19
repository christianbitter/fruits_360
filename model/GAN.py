import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

from .Model import Model


class FruitGAN(Model):
    def __init__(self, params_fp: str, verbose: bool = False):
        Model.__init__(FruitGAN, self).__init__(params_fp, "Fruit360GAN", verbose=verbose)

    @staticmethod
    def _build_generator(inputs, y_labels, image_size):
        """Build a Generator Model

        Inputs are concatenated before Dense layer.
        Stack of BN-ReLU-Conv2DTranpose to generate fake images.
        Output activation is sigmoid instead of tanh in orig DCGAN.
        Sigmoid converges easily.

        # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        y_labels (Layer): Input layer for one-hot vector to condition the inputs
        image_size: Target size of one side (assuming square image)

        # Returns
        Model: Generator Model
        """
        image_resize = image_size // 4
        # network parameters
        kernel_size = 5
        layer_filters = [128, 64, 32, 1]
        x = tf.keras.layers.concatenate([inputs, y_labels], axis=1)
        x = tf.keras.layers.Dense(image_resize * image_resize * layer_filters[0])(x)
        x = tf.keras.layers.Reshape((image_resize, image_resize, layer_filters[0]))(x)
        for filters in layer_filters:
            # first two convolution layers use strides = 2
            # the last two use strides = 1
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding='same')(x)

        x = tf.keras.layers.Activation('sigmoid')(x)
        # input is conditioned by y_labels
        generator = tf.keras.Model([inputs, y_labels], x, name='generator')
        return generator

    @staticmethod
    def _build_discriminator(inputs, y_labels, image_size):
        """Build a Discriminator Model

        Inputs are concatenated after Dense layer. Stack of LeakyReLU-Conv2D to discriminate real from fake.
        The network does not converge with BN so it is not used here unlike in DCGAN paper.

        # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        y_labels (Layer): Input layer for one-hot vector to condition the inputs
        image_size: Target size of one side (assuming square image)

        # Returns
        Model: Discriminator Model
        """
        kernel_size = 5
        layer_filters = [32, 64, 128, 256]
        x = inputs
        y = tf.keras.layers.Dense(image_size * image_size)(y_labels)
        y = tf.keras.layers.Reshape((image_size, image_size, 1))(y)
        x = tf.keras.layers.concatenate([x, y])

        for filters in layer_filters:
            # first 3 convolution layers use strides = 2
            # last one uses strides = 1
            if filters == layer_filters[-1]:
                strides = 1
            else:
                strides = 2
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding='same')(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        # input is conditioned by y_labels
        discriminator = tf.keras.Model([inputs, y_labels], x, name='discriminator')
        return discriminator

    def _parse_tfrecord(self, tfrecord_proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        features = {'label': tf.FixedLenFeature([], tf.int64),
                    'image_shape': tf.FixedLenFeature([], tf.string),
                    'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(tfrecord_proto, features)

        label = parsed_features['label']
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        shape = tf.decode_raw(parsed_features['image_shape'], tf.uint8)
        return image, label

    def _global_preprocess(self, data, label):
        if data is None:
            raise ValueError("_global_preprocess - data missing")
        if label is None:
            raise ValueError("_global_preprocess - label missing")
        data = tf.cast(data, dtype=tf.float32)
        data = data / 255.
        label = tf.cast(label, dtype=tf.int64)
        return data, label

    def import_data(self):
        if self.verbose:
            print("Loading tfrecord-train: {}".format(self.tfrecord_train_fp))
        train_dataset = tf.data.TFRecordDataset(self.tfrecord_train_fp)
        train_dataset = train_dataset.map(self._parse_tfrecord, num_parallel_calls=self.no_parallel_units)
        train_dataset = train_dataset.map(self._global_preprocess, num_parallel_calls=self.no_parallel_units)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(self.shuffle_buffer)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(self.batch_size)
        iterator = train_dataset.make_one_shot_iterator()

        image, label = iterator.get_next()
        self.data_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        self.label_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        self.training_set = train_dataset

        with open(self.training_fp, 'r') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            self.no_training_instances = sum(1 for _ in r)

        with open(self.testing_fp, 'r') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            self.no_testing_instances = sum(1 for _ in r)

        if self.verbose:
            print("Training Instances: {}/ Testing Instances: {}".format(self.no_training_instances, self.no_testing_instances))

        self.steps_per_epoch = math.ceil(self.no_training_instances / self.batch_size)

    def summary(self, what: str = 'full'):
        w = what.lower()
        if w == 'discriminator':
            print("Discriminator: {0}\r\n".format(self.discriminator))
        elif w == 'generator':
            print("Discriminator: {0}\r\n".format(self.generator))
        elif w == 'full':
            print("Discriminator: {0}\r\n".format(self.discriminator))
            print("Discriminator: {0}\r\n".format(self.generator))
        else:
            raise ValueError("summary - unknown option")