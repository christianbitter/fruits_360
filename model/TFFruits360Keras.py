import os
import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from .Params import Params
from PIL import Image
from .common import vgg_layer, fruit_classes


class TFFruits360(object):
    def __init__(self, train_fp=None, test_fp=None,
                 params_fp=None,
                 checkpoint_dir="checkpoint",
                 log_dir="logs",
                 verbose=False):
        self.tfrecord_train_fp = train_fp
        self.tfrecord_test_fp  = test_fp
        self.model_name = "FruitClassifier"
        self.params_fp = params_fp
        self.batch_size = None
        self.no_epochs = None
        self.learning_rate = None
        self.optimizer_name = None
        self._load_params()
        self.shuffle_buffer = self.batch_size
        # self.learning_rate = learning_rate
        self.image_width = 100
        self.image_height = 100
        self.no_image_channels = 3
        self.log_dir = log_dir
        self.checkpoint_dir_fp = checkpoint_dir
        self.steps_per_epoch = math.ceil(5159 / self.batch_size)  # hard fail
        self.image_iter = None
        self.label_iter = None
        self.model_input = None
        self.model_output = None
        self.training_set = None
        self.model = None
        self.verbose = verbose
        self.class_ids = fruit_classes()

    def _load_params(self):
        if self.params_fp and os.path.exists(self.params_fp):
            # now update where possible
            # if self.verbose:
            #     print("_load_params - loading parameters from json")
            self.params = Params.create(json_file_path = self.params_fp)
        else:
            self.params = Params({
                "learning_rate": 3e-1,
                "batch_size": 1,
                "no_epochs": 1,
                "optimizer": "rmsprop"
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
        self.no_epochs = self.params.params['no_epochs']
        self.learning_rate = self.params.params['learning_rate']
        self.optimizer_name = self.params.params['optimizer']
        self.shuffle_buffer = self.batch_size
        self.tfrecord_train_fp = self.params.params['training_tfrecord_fp']
        self.tfrecord_test_fp  = self.params.params['testing_tfrecord_fp']
        self.training_fp = self.params.params['training_fp']
        self.testing_fp = self.params.params['testing_fp']
        self.checkpoint_fp = self.params.params['checkpoint_fp']
        self.no_training_instances = -1
        self.no_testing_instances  = -1
        self.log_dir = self.params.params['log_fp']

    def _global_preprocess(self, data, label):
        data = tf.cast(data, dtype=tf.float32)
        data = data / 255.
        label = tf.cast(label, dtype=tf.int64)
        return (data, label)

    def _tfrecord_parse_function(self, tfrecord_proto):
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
        train_dataset = tf.data.TFRecordDataset(self.tfrecord_train_fp)
        train_dataset = train_dataset.map(self._tfrecord_parse_function, num_parallel_calls=8)
        train_dataset = train_dataset.map(self._global_preprocess, num_parallel_calls=8)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(self.shuffle_buffer)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(self.batch_size)
        iterator = train_dataset.make_one_shot_iterator()
        #
        image, label = iterator.get_next()
        self.image_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        self.label_iter = tf.one_hot(label, len(self.classes()))
        self.training_set = train_dataset

    def build(self):
        use_input_dropout = False
        if 'input_dropout' in self.params:
            use_input_dropout = True
            input_dropout_rate = self.params['input_dropout']
            if input_dropout_rate <= 0:
                use_input_dropout = False

        if 'layer_dropout' in self.params:
            use_layer_dropout = True
            layer_dropout_rate = self.params['layer_dropout']
            if layer_dropout_rate <= 0:
                use_layer_dropout = False

        with tf.name_scope('input'):
            self.model_input = tf.keras.layers.Input(shape=(self.image_width, self.image_height, self.no_image_channels))
            x = self.model_input
            if use_input_dropout:
                x = keras.layers.Dropout(rate=input_dropout_rate)(x)

        with tf.name_scope('layer1'):
            x = vgg_layer(filter_size=32, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer2'):
            x = vgg_layer(filter_size=64, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer3'):
            x = vgg_layer(filter_size=256, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer4'):
            x = vgg_layer(filter_size=512, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer5'):
            x = vgg_layer(filter_size=1024, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope("output"):
            x = keras.layers.Flatten()(x)

            x = keras.layers.Dense(1024)(x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

            model_output = tf.keras.layers.Dense(len(self.classes()), activation='softmax')(x)
            self.model_output = model_output

    def prepare(self, model_weights_fp:str=None):
        self.model = tf.keras.models.Model(inputs=self.model_input, outputs=self.model_output)
        optimizer = {
            "rmsprop": tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            "adam": tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        }.get(self.optimizer_name, None)
        if not optimizer:
            raise ValueError("TFFruit360Keras - unknown optimizer '{0}'.".format(self.optimizer_name))

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

        self.model.compile(optimizer=optimizer, loss=fn_loss_name, metrics=v_metrics)

    def train(self):
        if not self.training_set:
            raise ValueError("train - training data set not loaded")

        callbacks = []
        if self.log_dir:
            if self.verbose:
                print("Enabling Keras Tensorboard Callback: {0}".format(self.log_dir))
            kcb = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False)
            callbacks.append(kcb)

        if self.checkpoint_dir_fp:
            save_best = False
            checkpoint_path = "checkpoint/cp_tffruit360_{0}_{1}_{2}_{3}.ckpt".format(self.learning_rate, self.batch_size, self.no_epochs, "{epoch:04d}")
            checkpoint_dir = os.path.dirname(checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=save_best, verbose=1)
            callbacks.append(cp_callback)

        return self.model.fit(
            self.image_iter, self.label_iter,
            epochs=self.no_epochs, steps_per_epoch=self.steps_per_epoch,
            verbose=True, callbacks=callbacks)

    def preprocess_input(self, image:Image):
        x = np.asarray(image, np.uint8)
        x = np.array(x).reshape((1, self.image_height, self.image_width, self.no_image_channels))
        x = tf.cast(x, dtype=tf.float32)
        x = x / 255.
        return x

    def predict(self, image:Image) -> dict:
        if not image:
            raise ValueError("predict - image not provided")
        if not self.model:
            raise ValueError("predict - cannot predict on untrained model")
        x = self.preprocess_input(image)
        s = self.model.predict(x, steps=1, verbose=True)
        ci = np.argmax(s)

        return {
            "class_id": ci,
            "class_name": self.classid_to_class(ci),
            "predictions": s
        }

    def from_checkpoint(self, checkpoint_fp):
        self.build()
        self.prepare(model_weights_fp=checkpoint_fp)

    def evaluate(self):
        test_dataset = tf.data.TFRecordDataset(self.tfrecord_test_fp)
        test_dataset = test_dataset.map(self._tfrecord_parse_function, num_parallel_calls=8)
        test_dataset = test_dataset.map(self._global_preprocess, num_parallel_calls=8)
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.prefetch(self.batch_size)
        iterator = test_dataset.make_one_shot_iterator()
        #
        image, label = iterator.get_next()
        image_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        label_iter = tf.one_hot(label, len(self.classes()))

        no_testing_steps = 100

        return self.model.evaluate(x=image_iter, y=label_iter,  steps=no_testing_steps, verbose=1)

    def class_to_id(self, lbl_name: str) -> int:
        return self.class_ids.get(lbl_name, -1)

    def classid_to_class(self, class_id:int) -> str:
        c_name = [m for m in self.class_ids if self.class_ids[m] == class_id]
        return c_name[0]

    def classes(self):
        return [key for key in self.class_ids.keys()]

    def summary(self):
        print(self.model.summary())

    def plot(self, what:str=None):
        pass