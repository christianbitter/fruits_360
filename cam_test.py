# TODO: this has to go into it's own model, using the pre-trained ones4
import os
from PIL import Image
import numpy as np
from model.TFFruits360Keras import TFFruits360
from model.ClassActivationMaps import ClassActivationMap
from data.data_prepare import dir_to_class
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy

# we load a pre-trained fruits model
verbose=True
plot_image=True

cam = ClassActivationMap(params_fp="D:/Users/gfrsa/python/tensorflow/fruits_360/experiments/base_model/params_cam.json", verbose=verbose)
cam.from_checkpoint(checkpoint_fp="D:/Users/gfrsa/python/tensorflow/fruits_360/checkpoint/cp_tffruit360_0.0001_30_10_0010.ckpt")

image_fp = "F:/data/fruits_360/fruits-360/Test/Apple Braeburn/3_100.jpg"
i = Image.open(image_fp)
lbl = os.path.basename(os.path.dirname(image_fp))
lbl = dir_to_class(lbl)

r = cam.predict(i)
a_i = r["cam"]
predicted_clsname = r["class_name"]
if plot_image:
    plt.subplot(1, 2, 1)
    plt.imshow(i, alpha=.8)
    plt.imshow(a_i, cmap='jet', alpha=.5)
    plt.title("Class activation map")
    plt.subplot(1, 2, 2)
    plt.imshow(i)
    plt.title("Label - {0} <-> Predicted - {1}".format(lbl, predicted_clsname))
    plt.show()
