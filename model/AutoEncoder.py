import tensorflow as tf
import numpy as np
import os
import math
import csv
from .Params import Params
from .common import fruit_classes
from PIL import Image
import matplotlib.pyplot as plt

class FruitAutoEncoder(object):
    def __init__(self, params_fp, verbose=False):
        self.tfrecord_train_fp = None
        self.tfrecord_test_fp  = None
        self.training_fp = None
        self.testing_fp = None
        self.model_name = "FruitAutoEncoder"
        self.log_dir = None
        self.checkpoint_fp = None
        self.params_fp = params_fp
        self.verbose = verbose
        self.image_width = 100
        self.image_height = 100
        self.no_hidden_dimensions = 30
        self.no_image_channels = 3
        # our filters/ conv layers have to result in even numbered dimensions e.g., (100, 100) => (50, 50) => (25, 25) is the deepest
        self.filter_size = [16, 32]
        self.conv_kernel = (3, 3)
        self.filter_activation = 'relu'
        self.no_parallel_units = 8

        self.batch_size = None
        self.shuffle_buffer = None
        self.no_epochs = None
        self.learning_rate = None
        self.optimizer_name = None
        self.steps_per_epoch = None

        self._load_params()

    def _load_params(self):
        if self.params_fp and os.path.exists(self.params_fp):
            # now update where possible
            self.params = Params.create(json_file_path = self.params_fp)
        else:
            self.params = Params({
                "learning_rate": 3e-1,
                "batch_size": 1,
                "no_epochs": 1,
                "optimizer": "rmsprop",
                "checkpoint_fp": "checkpoint",
                "log_fp": "logs"
            })

        if 'learning_rate' not in self.params:
            raise ValueError("_load_params - learning_rate not present")
        if 'batch_size' not in self.params:
            raise ValueError("_load_params - batch_size not present")
        if 'no_epochs' not in self.params:
            raise ValueError("_load_params - no_epochs not present")
        if 'optimizer' not in self.params:
            raise ValueError("_load_params - optimizer not present")
        if 'training_tfrecord_fp' not in self.params:
            raise ValueError("_load_params - training_tfrecord_fp not present")
        if 'testing_tfrecord_fp' not in self.params:
            raise ValueError("_load_params - testing_tfrecord_fp not present")
        if 'training_fp' not in self.params:
            raise ValueError("_load_params - training_fp not present")
        if 'testing_fp' not in self.params:
            raise ValueError("_load_params - testing_fp not present")
        if 'log_fp' not in self.params:
            raise ValueError("_load_params - log_fp not present")
        if 'checkpoint_fp' not in self.params:
            raise ValueError("_load_params - checkpoint_fp not present")

        self.batch_size = self.params.params['batch_size']
        self.shuffle_buffer = self.batch_size
        self.no_epochs = self.params.params['no_epochs']
        self.learning_rate = self.params.params['learning_rate']
        self.optimizer_name = self.params.params['optimizer']
        self.tfrecord_train_fp = self.params.params['training_tfrecord_fp']
        self.tfrecord_test_fp  = self.params.params['testing_tfrecord_fp']
        self.training_fp = self.params.params['training_fp']
        self.testing_fp = self.params.params['testing_fp']
        self.checkpoint_fp = self.params.params['checkpoint_fp']
        self.no_training_instances = -1
        self.no_testing_instances  = -1
        self.log_dir = self.params.params['log_fp']


    def _global_preprocess(self, data, label):
        if data is None:
            raise ValueError("_global_preprocess - data missing")
        if label is None:
            raise ValueError("_global_preprocess - label missing")
        data = tf.cast(data, dtype=tf.float32)
        data = data / 255.
        label = tf.cast(label, dtype=tf.int64)
        return data, label

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

    def build(self):
        with tf.name_scope('input'):
            self.model_input = tf.keras.layers.Input(shape=(self.image_width, self.image_height, self.no_image_channels)) # tensor=self.image_iter)
            x = self.model_input

        # a series of convolutional down filters
        with tf.name_scope('encode'):
            for filter_size in self.filter_size:
                x = tf.keras.layers.Conv2D(filters=filter_size, kernel_size=self.conv_kernel, strides=(2,2),
                                           padding='same', activation=self.filter_activation)(x)
        unfold_conv = x

        with tf.name_scope('embedding'):
            # a dense layer
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(units=self.no_hidden_dimensions)(x)
            latent = x
            self.encoder = tf.keras.Model(self.model_input, latent, name='encoder')

        # unfold the last conv into a dense with the appropriate number of neurons
        latent_inputs = tf.keras.layers.Input(shape=(self.no_hidden_dimensions,), name='decoder_input')
        shape = unfold_conv.shape
        x = tf.keras.layers.Dense(units=shape[1] * shape[2] * shape[3])(latent_inputs)
        # from vector to suitable shape for transposed conv
        x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        for filter_size in self.filter_size[::-1]:
            # a series of convolutional up filters
            x = tf.keras.layers.Conv2DTranspose(filters=filter_size, kernel_size=self.conv_kernel, strides=(2, 2),
                                                padding='same', activation=self.filter_activation)(x)

        with tf.name_scope("output"):
            x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=self.conv_kernel,
                                                activation='sigmoid', padding='same', name='decoder_output')(x)
            model_output = x
            print("Shape: {}".format(model_output.shape))
            self.model_output = model_output
            self.decoder = tf.keras.Model(latent_inputs, self.model_output, name='decoder')

    def prepare(self, model_weights_fp: str=None):
        self.autoencoder = tf.keras.models.Model(inputs=self.model_input, outputs=self.decoder(self.encoder(self.model_input)), name='autoencoder')
        optimizer = {
            "rmsprop": tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            "adam": tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        }.get(self.optimizer_name, None)
        if not optimizer:
            raise ValueError("prepare - unknown optimizer '{0}'.".format(self.optimizer_name))

        if model_weights_fp:
            weights_fp_idx = "{0}.index".format(model_weights_fp)

            if os.path.exists(weights_fp_idx):
                self.model.load_weights(model_weights_fp)
            else:
                raise ValueError("TFFruit360Keras - weights file specified but does not exist")

        if 'loss_function' in self.params:
            fn_loss_name = self.params['loss_function']
        else:
            fn_loss_name = 'mean_squared_error'

        if 'performance_metric' in self.params:
            v_metrics = self.params['performance_metric']
        else:
            v_metrics = ['acc', 'mae']
        self.autoencoder.compile(optimizer=optimizer, loss=fn_loss_name, metrics=v_metrics)

    def summary(self, what:str='full'):
        {
            'encoder': print(self.encoder.summary()),
            'decoder': print(self.decoder.summary()),
            'full': print(self.autoencoder.summary())
        }.get(what)

    def __plot_embedding(self):
        print("__plot_embedding")
        # display a 2D plot of the digit classes in the latent space
        test_dataset = tf.data.TFRecordDataset(self.tfrecord_test_fp)
        test_dataset = test_dataset.map(self._parse_tfrecord, num_parallel_calls=8)
        test_dataset = test_dataset.map(self._global_preprocess, num_parallel_calls=8)
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.prefetch(self.batch_size)

        no_points = math.ceil(self.no_testing_instances / self.batch_size)
        # we could also use the initializer iterator and call initialize two times
        iterator = test_dataset.make_one_shot_iterator()
        _, label = iterator.get_next()
        v_label = []
        i = 0
        with tf.Session() as sess:
            while i < no_points:
                l = sess.run(label)
                v_label.extend(l)
                i += 1
        labels = v_label
        iterator = test_dataset.make_one_shot_iterator()
        train_data, train_label = iterator.get_next()
        train_data = tf.reshape(train_data, [-1, self.image_width, self.image_height, self.no_image_channels])

        print("No testing instances: {}/ Batch Size: {}/ No Points: {}".format(self.no_testing_instances, self.batch_size, no_points))
        z = self.encoder.predict(train_data, steps=no_points)
        plt.figure(figsize=(12, 10))
        plt.scatter(z[:, 0], z[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("embedding dim 0")
        plt.ylabel("embedding dim 1")
        plt.title("2D Embedding of Eutoencoded Fruit Images")
        plt.show()

    def plot(self, what:str=None):
        print("plot ({})".format(what))
        {
            'embedding': self.__plot_embedding()
        }.get(what)

    def train(self):
        if not self.training_set:
            raise ValueError("train - training data set not loaded")

        callbacks = []
        if self.log_dir:
            if self.verbose:
                print("Enabling Keras Tensorboard Callback: {0}".format(self.log_dir))
            kcb = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False)
            callbacks.append(kcb)

        if self.checkpoint_fp:
            save_best = False
            checkpoint_path = "{}/cp_{}_{}_{}_{}_{}.ckpt".format(self.checkpoint_fp, self.model_name, self.learning_rate, self.batch_size, self.no_epochs, "{epoch:04d}")
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=save_best, verbose=1)
            callbacks.append(cp_callback)

        # in the auto-encoder case, data is label and data
        if self.verbose:
            print("Number of Training Instances: {}/ Number of Epochs: {} / Steps Per Epoch: {}".format(self.no_training_instances, self.no_epochs, self.steps_per_epoch))
        return self.autoencoder.fit(
            self.data_iter, self.data_iter,
            epochs=self.no_epochs, steps_per_epoch=self.steps_per_epoch,
            verbose=True, callbacks=callbacks)

    def evaluate(self):
        pass

    def predict(self):
        pass