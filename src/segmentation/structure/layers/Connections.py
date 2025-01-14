from keras.src.layers import *
from tensorflow.keras.layers import *

from src.segmentation.structure.layers.Convolutional import convolutional_layer_2D, convolutional_layer_3D
from src.segmentation.structure.layers.SnE import squeeze_excitation_layer


# Configuration - 1
# Dilated layer-based skip-connection between encoder and decoder layer.
def dilated_skip_connection(x, mode="3D", filters=8, dilated_params=None):
    if dilated_params["dilation"] is not None:
        dilation_rates = dilated_params["dilation"]
    else:
        dilation_rates = [1, 2, 4, 8]

    for i in dilation_rates:
        dilated_params['dilation'] = i
        if mode == "3D":
            x = convolutional_layer_3D(x, 1, filters, **dilated_params)
        else:
            x = convolutional_layer_2D(x, 1, filters, **dilated_params)

    return x


# Configuration - 2
# Dilated layer-based residual_skip-connection between encoder and decoder layer.
def dilated_skip_res_connection(x, mode="3D", filters=8, dilated_params=None):
    return Add()([x, dilated_skip_connection(x, mode, filters, dilated_params)])


# Configuration - 3
# Dilated Layer processing block followed by a sne-block between encoder and decoder layer.
def dilated_sne_skip_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return sne_skip_connection(dilated_skip_connection(x, mode, filters, dilated_params),
                               mode, **sne_params)


#   TO-DO: Add in configurations
# Configuration - 3A
# sne block followed by a dilated layer between encoder and decoder layer.
def sne_dilated_skip_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return dilated_skip_connection(sne_skip_connection(x, mode, **sne_params),
                                   mode, filters, dilated_params)


# Configuration - 4
# Dilated Layer processing block followed by a  between encoder and decoder layer.
def dilated_sne_skip_res_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return Add()([x, dilated_sne_skip_connection(x, mode, filters, dilated_params, sne_params)])


#   TO-DO: Add in configurations
# Configuration - 4A
# Dilated Layer processing block followed by a  between encoder and decoder layer.
def sne_dilated_skip_res_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return Add()([x, sne_dilated_skip_connection(x, mode, filters, dilated_params, sne_params)])


# Configuration - 5
# SnE-based skip connection between encoder and decoder layer.
def sne_skip_connection(x, mode="3D", sne_params=None):
    return squeeze_excitation_layer(x, mode, **sne_params)


# Configuration - 6
# SnE-based skip connection with residual connection.
def sne_skip_res_connection(x, mode="3D", sne_params=None):
    return Add()([x, squeeze_excitation_layer(x, mode, **sne_params)])


#   TO-DO: Add in configurations
# Configuration - 7
# SnE-based and dilated connection is applied separately and then added.
def sne_and_dilated_skip_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return Add()[sne_skip_connection(x, mode, **sne_params), dilated_skip_connection(x, mode, filters, dilated_params)]


#   TO-DO: Add in configurations
# Configuration - 8
# SnE-based and dilated connection is applied separately and then added.
def sne_and_dilated_skip_res_connection(x, mode="3D", filters=8, dilated_params=None, sne_params=None):
    return Add()[x, sne_skip_connection(x, mode, **sne_params), dilated_skip_connection(x, mode, filters, dilated_params)]
