import os
from PIL import Image
import numpy as np
from model.TFFruits360Keras import TFFruits360
from model.AutoEncoder import FruitAutoEncoder
from model.GAN import FruitGAN
from data.data_prepare import dir_to_class
import matplotlib.pyplot as plt
import argparse

# example args:
# python evaluate.py --mode=classify --action=train --verbose --tf-train="F:\data\fruits_360\fruits-360\tf_data\train.rec" --tf-test="F:\data\fruits_360\fruits-360\tf_data\test.rec" --params="D:\Users\gfrsa\python\tensorflow\fruits_360\experiments\base_model\params_classification.json"
# python evaluate.py --mode=classify --action=eval --verbose --tf-test="F:\data\fruits_360\fruits-360\tf_data\test.rec" --params="D:\Users\gfrsa\python\tensorflow\fruits_360\experiments\base_model\params.json" --checkpoint="D:/Users/gfrsa/python/tensorflow/fruits_360/checkpoint/cp_tffruit360_0.0001_20_10_0008.ckpt"
# python evaluate.py --mode=classify --action=predict --verbose --params="D:\Users\gfrsa\python\tensorflow\fruits_360\experiments\base_model\params.json" --checkpoint="D:/Users/gfrsa/python/tensorflow/fruits_360/checkpoint/cp_tffruit360_0.0001_20_10_0008.ckpt" --input "F:\data\fruits_360\fruits-360\Test\Apple Braeburn\3_100.jpg"

def cli():
    description = "The Fruit360 Data Preparation tool allows you to "
    parser = argparse.ArgumentParser(prog="Evaluate", description=description)

    parser.add_argument("--mode", help="The type of model to use", required=True, default="classify", dest="mode", type=str, choices=["classify", "autoencode"])
    parser.add_argument("--action", help="The action to be performed (train, eval, predict)", required=True, dest="action", type=str, choices=["train", "eval", "predict"])
    parser.add_argument("--tf-train", help="Full path to the train tfrecord file.", required=False, nargs=1, dest="train_fp", type=str)
    parser.add_argument("--tf-test", help="Full path to the validation tfrecord file.", required=False, nargs=1, dest="test_fp", type=str)
    parser.add_argument("--checkpoint", help="Full path to the trained model checkpoint file.", required=False, nargs=1, dest="checkpoint_fp", type=str)
    parser.add_argument("--input", help="Full path to images to be run predictions for.", required=False, nargs=1, dest="input_fp", type=str)
    parser.add_argument("--params", help="Full path to the parameter json file.", required=False, nargs=1, dest="params_fp", type=str)
    parser.add_argument("--verbose", help="Should verbose mode be switched on.", required=False, action='store_true', default=False)

    return parser

def plot_training_history(training_history, args, verbose:bool=False):
    # TODO: evaluate the params to see if the validation metrics are defined
    # Plot training & validation accuracy values
    plt.figure(1)
    plt.subplot(121)
    plt.plot(training_history.history['acc'])
    # plt.plot(training_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(training_history.history['loss'])
    # plt.plot(training_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


def train(mode:str, args, verbose:bool=False):
    if verbose:
        print("Training: ({})".format(mode))

    if 'train_fp' not in args:
        raise ValueError("train - tf-train not defined in command line arguments")
    if 'test_fp' not in args:
        raise ValueError("train - tf-test not defined in command line arguments")
    if 'params_fp' not in args:
        raise ValueError("train - params not defined in command line arguments")

    train_fp = args["train_fp"][0]
    test_fp = args["test_fp"][0]
    params_fp = args["params_fp"][0]

    model = None
    if mode == 'classify':
        model = TFFruits360(train_fp=train_fp, test_fp=test_fp, params_fp=params_fp, verbose=verbose)
    if mode == 'autoencode':
        model = FruitAutoEncoder(params_fp=params_fp, verbose=verbose)
    if mode == 'gan':
        model = FruitGAN(params_fp=params_fp, verbose=verbose)

    model.import_data()
    model.build()
    model.prepare()
    if verbose:
        print(model.summary())

    training_history = model.train()
    plot_training_history(training_history=training_history, args=args, verbose=verbose)

    if mode == 'autoencode':
        model.plot(what='embedding')

    return training_history


def eval(mode:str, args, verbose:bool=False):
    if verbose:
        print("Evaluation")

    if 'checkpoint_fp' not in args:
        raise ValueError("train - checkpoint_fp not defined in command line arguments")
    if 'test_fp' not in args:
        raise ValueError("train - tf-test not defined in command line arguments")

    test_fp = args["test_fp"][0]
    params_fp = args["params_fp"][0]
    checkpoint_fp = args["checkpoint_fp"][0]
    model = TFFruits360(train_fp=None, test_fp=test_fp, params_fp=params_fp)
    model.from_checkpoint(checkpoint_fp=checkpoint_fp)
    evaluation_score = model.evaluate()
    print("Evaluation Score Loss/Accuracy: {0}".format(evaluation_score))
    return None

def predict(mode:str, args, verbose:bool=False):
    if verbose:
        print("Prediction")
    if 'checkpoint_fp' not in args:
        raise ValueError("train - checkpoint_fp not defined in command line arguments")
    if 'input_fp' not in args:
        raise ValueError("train - input_fp not defined in command line arguments")

    params_fp = args["params_fp"][0]
    checkpoint_fp = args["checkpoint_fp"][0]
    input_fp = args["input_fp"][0]
    model = TFFruits360(train_fp=None, test_fp=None, params_fp=params_fp)
    model.from_checkpoint(checkpoint_fp=checkpoint_fp)
    image_fp  = input_fp
    i = Image.open(image_fp)

    lbl = os.path.basename(os.path.dirname(image_fp))
    lbl = dir_to_class(lbl)
    # test prediction
    pred = model.predict(i)

    plt.imshow(i)
    plt.title("Label - {0} <-> Predicted - {1}".format(lbl, pred))
    plt.show()

    return lbl == pred

def main():
    parser = cli()
    parsed_args = vars(parser.parse_args())
    verbose = parsed_args["verbose"]

    if verbose:
        print("Verbose Mode Switched on")

    action = parsed_args["action"]
    mode   = parsed_args["mode"]

    return {
        "train": lambda args: train(mode=mode, args=parsed_args, verbose=verbose),
        "eval": lambda args: eval(mode, args, verbose=verbose),
        "predict": lambda args: predict(mode, args, verbose=verbose)
    }.get(action)(parsed_args)


if __name__ == '__main__':
    main()