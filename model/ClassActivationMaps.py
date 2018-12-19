import os
import scipy
import tensorflow as tf
import numpy as np
from tensorflow import keras
from .Params import Params
from PIL import Image
from .common import vgg_layer, fruit_classes
from .Model import Model

# we build the class activation model on the pre-trained 360 model
class ClassActivationMap(Model):
    def __init__(self, params_fp=None, verbose=False):
        super(ClassActivationMap, self).__init__(params_fp=params_fp, model='ClassActivationMap', verbose=verbose)

        self.image_width = None
        self.image_height = None
        self.no_image_channels = None
        self.class_ids = fruit_classes()

        self._load_params()

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
        pass

    def _load_params(self):
        if self.verbose:
            print("ClassActivationMap - _load_params")

        if 'image_width' in self.params:
            self.image_width = self.params['image_width']
        if 'image_height' in self.params:
            self.image_height = self.params['image_height']
        if 'no_image_channels' in self.params:
            self.no_image_channels = self.params['no_image_channels']
        if 'learning_rate' in self.params:
            self.learning_rate = self.params['learning_rate']
        if 'batch_size' in self.params:
            self.batch_size = self.params['batch_size']
        if 'no_epochs' in self.params:
            self.no_epochs = self.params['no_epochs']
        if 'optimizer' in self.params:
            self.optimizer_name = self.params['optimizer']

    def preprocess_input(self, image:Image):
        x = np.asarray(image, np.uint8)
        x = np.array(x).reshape((1, self.image_height, self.image_width, self.no_image_channels))
        x = tf.cast(x, dtype=tf.float32)
        x = x / 255.
        return x

    def predict(self, image: Image):
        if not image:
            raise ValueError("predict - image not provided")
        if not self.model:
            raise ValueError("predict - cannot predict on untrained model")

        x = self.preprocess_input(image)
        s = self.model.predict(x, steps=1, verbose=True)
        activations = self.activation_model.predict(x, steps=1)
        ci = np.argmax(s)
        cn = self.classid_to_class(ci)

        # let's get the layer weights
        w_i =self.W_dense_out[:, ci]
        # combine with activations - note that the number of weights, i.e. the inputs to the final dense layer
        # need to be the same as the outputs of the activation - here 1024
        a_i = activations.dot(w_i).reshape(self.activation_shape)
        f   = self.image_width / self.activation_shape[0]

        a_i = scipy.ndimage.zoom(a_i, f, order=1)

        return {
            "class_id": ci,
            "class_name": cn,
            "predictions": s,
            "cam": a_i
        }

    def from_checkpoint(self, checkpoint_fp):
        self.build()
        self.prepare(model_weights_fp=checkpoint_fp)

        self.final_activation_layer = self.model.get_layer('activation_4')  # 7,7,256
        # we turn this layer into a new model, so that we can get the activations as model output
        self.activation_model = tf.keras.Model(inputs=self.model.input, outputs=self.final_activation_layer.output)
        s = self.activation_model.output.shape
        self.activation_shape = (int(s[1]), int(s[2]))

        dense_out_layer = self.model.get_layer('dense_1')
        self.W_dense_out = dense_out_layer.get_weights()[0]  # 4000, 49

    def evaluate(self):
        pass

    def class_to_id(self, lbl_name: str) -> int:
        return self.class_ids.get(lbl_name, -1)

    def classid_to_class(self, class_id:int) -> str:
        c_name = [m for m in self.class_ids if self.class_ids[m] == class_id]
        return c_name[0]

    def classes(self):
        return [key for key in self.class_ids.keys()]

    def summary(self, what: str = 'full'):
        {
            'full': print("Model:{0}\r\nActivation_Model: {1}".format(self.model.summary(), self.activation_model.summary())),
            'activation': print("Activation_Model: {0}".format(self.activation_model.summary()))
        }.get(what)

    def plot(self, what: str=None):
        # TODO: plot the class activation map
        pass