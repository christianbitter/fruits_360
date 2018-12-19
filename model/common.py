from tensorflow import keras

def fruit_classes():
    return {
        'rambutan': 0,
        'tomato': 1,
        'huckleberry': 2,
        'mulberry': 3,
        'melon': 4,
        'carambula': 5,
        'avocado': 6,
        'pomegranata': 7,
        'apple': 8,
        'mandarine': 9,
        'cocos': 10,
        'pitahaya': 11,
        'walnut': 12,
        'nectarine': 13,
        'tangelo': 14,
        'mango': 15,
        'grapefruit': 16,
        'kaki': 17,
        'pepino': 18,
        'Cantaloupe': 19,
        'tamarillo': 20,
        'clementine': 21,
        'banana': 22,
        'cherry': 23,
        'lemon': 24,
        'papaya': 25,
        'plum': 26,
        'cantaloupe': 27,
        'lychee': 28,
        'passion fruit': 29,
        'strawberry': 30,
        'raspberry': 31,
        'orange': 32,
        'physalis': 33,
        'cactus fruit': 34,
        'grape': 35,
        'dates': 36,
        'apricot': 37,
        'guava': 38,
        'maracuja': 39,
        'quince': 40,
        'pineapple': 41,
        'limes': 42,
        'salak': 43,
        'kiwi': 44,
        'pear': 45,
        'granadilla': 46,
        'peach': 47,
        'kumquats': 48
    }


def vgg_layer(filter_size:int, input, padding:str='same', activation:str='relu'):
    x = input
    x = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding=padding)(x)
    x = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=activation)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x