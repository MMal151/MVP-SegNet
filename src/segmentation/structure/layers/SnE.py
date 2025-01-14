import tensorflow as tf
from tensorflow import add, multiply
from tensorflow.keras.layers import *
from keras.src.layers import *


from src.segmentation.structure.layers.Convolutional import convolutional_layer, dimension_reduction_layer, \
    upsampling_layer

CLASS_NAME = "[Layers/SnE]"


def sne_down_layer(x, mode="3D", filters=None, num_conv_blocks=None, conv_params=None, sne_params=None,
                   down_params=None):
    lgr = CLASS_NAME + "[sne_down_layer()]"

    assert filters > 0, f"{lgr}[Error]: Number of filters must be greater than 0. filters = [{filters}]"
    assert num_conv_blocks > 0, f"{lgr}[Error]: Number of convolutional blocks must be greater than 0. " \
                                f"num_conv_blocks = [{num_conv_blocks}]"

    y = x  # Saving the original state of the tensor as a residual connection.
    x = convolutional_layer(x, mode, filters, num_conv_blocks, **conv_params)
    x = squeeze_excitation_layer(x, mode, **sne_params)
    y = dimension_reduction_layer(y, filters, mode)  # Simple 1x1 Convolution
    x = y + x

    return x, convolutional_layer(x, mode, filters, 1,
                                  **down_params)  # Number of convolutional blocks = 1 for down-sampling layer.


def sne_up_layer(x, rc, mode="3D", filters=None, num_conv_blocks=1, use_transpose=False, sne_params=None,
                 conv_params=None, up_params=None):
    x = upsampling_layer(x, mode, filters, use_transpose, up_params)

    if x.shape[1:-1] != rc.shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - rc.shape[1])
        crop_height = max(0, x.shape[2] - rc.shape[2])
        crop_width = max(0, x.shape[3] - rc.shape[3])

        # Apply cropping
        x = Cropping3D(cropping=((0, crop_depth), (0, crop_height), (0, crop_width)))(x)

    y = x  # Saving the state of the input tensor to be added later
    x = Concatenate(axis=-1)([x, rc])
    x = convolutional_layer(x, mode, filters, num_conv_blocks, **conv_params)
    x = squeeze_excitation_layer(x, mode, **sne_params)
    y = dimension_reduction_layer(y, filters, mode)  # Simple 1x1 Convolution
    x = y + x

    return x


def squeeze_excitation_layer(x, mode="3D", ratio=2, use_res=True):
    if mode == "2D":
        return squeeze_excitation_block_2D(x, ratio=ratio, use_res=use_res)
    else:
        return squeeze_excitation_block_3D(x, ratio=ratio, use_res=use_res)


def squeeze_excitation_block_2D(x, ratio=2, use_res=True):
    y = x
    org_shape = x.shape

    # Squeeze
    x = GlobalAveragePooling2D()(x)

    x = Dense(units=(org_shape[-1] / ratio), activation='relu')(x)  # Original Paper
    x = Dense(org_shape[-1], activation='sigmoid')(x)
    x = tf.reshape(x, [-1, 1, 1, org_shape[-1]])

    # Scaling
    x = multiply([y, x])

    if use_res:
        y = Conv2D(org_shape[-1], kernel_size=1, strides=1, padding='same')(y)
        y = SpatialDropout2D(0.1)(y)  # TO-DO make it configurable
        y = BatchNormalization()(y)

        x = add([y, x])
        x = ReLU()(x)
    return x


# Squeeze and Excitation Block with residual connection. [SE-WRN]
def squeeze_excitation_block_3D(x, ratio=2, use_res=True):
    y = x
    org_shape = x.shape
    temp = int(org_shape[-1] / ratio)

    # Squeeze
    x = GlobalAveragePooling3D()(x)

    # Excitation
    x = Dense(int(org_shape[-1] / ratio), activation='relu')(x)  # Original Paper
    x = Dense(org_shape[-1], activation='sigmoid')(x)
    x = tf.reshape(x, [-1, 1, 1, 1, org_shape[-1]])

    # Scaling
    x = multiply([y, x])

    if use_res:
        y = Conv3D(org_shape[-1], kernel_size=1, strides=1, padding='same')(y)
        y = SpatialDropout3D(0.1)(y)
        y = BatchNormalization()(y)

        x = add([y, x])
        x = ReLU()(x)

    return x
