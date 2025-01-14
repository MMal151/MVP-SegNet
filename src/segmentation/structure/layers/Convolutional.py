#Rmove
from keras.src.layers import *
from keras_unet_collection.activations import GELU, Snake
from tensorflow.keras.layers import *

CLASS_NAME = "[Layers/Convolutional]"


def Activation(x, act):
    lgr = CLASS_NAME + "[Activation()]"
    if act == 'leakyrelu':
        return LeakyReLU()(x)
    elif act == 'relu':
        return ReLU()(x)
    elif act == 'prelu':
        return PReLU()(x)
    elif act == 'gelu':
        return GELU(trainable=True)(x)
    elif act == 'snake':
        return Snake(trainable=True)(x)
    else:
        print(f"{lgr}[Warning]: Invalid value given as activation function, using ReLU as default value.")
        return ReLU()(x)


def Normalisation(x, norm):
    lgr = CLASS_NAME + "[Normalisation()]"
    if norm == 'batch':
        return BatchNormalization()(x)
    elif norm == 'layer':
        return LayerNormalization()(x)
    else:
        print(
            f"{lgr}[Warning]: Invalid value given as normalisation function, using Batch Normalisation as default value.")
        return BatchNormalization()(x)


def convolution_with_downsampling_layer(x, mode="3D", filters=8, num_of_iter=1, conv_params=None, down_params=None):
    lgr = CLASS_NAME + "[convolution_with_downsampling_layer()]"

    if mode == '2D':
        x = convolutional_layer_2D(x, num_of_iter, filters, **conv_params)
        return x, convolutional_layer_2D(x, 1, filters,
                                         **down_params)  # Number of iterations for down-sampling will always be 1.
    else:
        x = convolutional_layer_3D(x, num_of_iter, filters, **conv_params)
        return x, convolutional_layer_3D(x, 1, filters,
                                         **down_params)  # Number of iterations for down-sampling will always be 1.


def upsampling_layer(x, mode="3D", filters=8, use_transpose=False, up_params=None):
    if use_transpose:
        x = transpose_convolution(x, mode, filters, **up_params)
    else:
        if mode == "2D":
            x = UpSampling2D()(x)
        else:
            x = UpSampling3D()(x)

    return x


def convolution_with_upsampling_layer(x, rc, mode, filters, num_conv_blocks=1, use_transpose=True, conv_params=None,
                                      up_params=None):
    x = upsampling_layer(x, mode, filters, use_transpose, up_params)
    x = cropping_layer(x, rc.shape, mode)
    y = x  # Saving the state of the input tensor to be added later
    x = Concatenate(axis=-1)([x, rc])

    x = convolutional_layer(x, mode, filters, num_conv_blocks, **conv_params)
    y = dimension_reduction_layer(y, filters, mode)  # Simple 1x1 Convolution
    x = y + x

    return x


def cropping_layer(x, desired_shape, mode="3D"):
    if mode == "2D":
        return cropping_2D(x, desired_shape)
    else:
        return cropping_3D(x, desired_shape)


def cropping_2D(x, desired_shape):
    if x.shape[1:-1] != desired_shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - desired_shape[1])
        crop_height = max(0, x.shape[2] - desired_shape[2])

        # Apply cropping
        return Cropping2D(cropping=((0, crop_depth), (0, crop_height)))(x)
    return x


def cropping_3D(x, desired_shape):
    if x.shape[1:-1] != desired_shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - desired_shape[1])
        crop_height = max(0, x.shape[2] - desired_shape[2])
        crop_width = max(0, x.shape[3] - desired_shape[3])

        # Apply cropping
        return Cropping3D(cropping=((0, crop_depth), (0, crop_height), (0, crop_width)))(x)

    return x


def convolutional_layer(x, mode="3D", filters=8, num_of_iter=1, kernel=3, stride=1, padding='same', activation='relu',
                        dilation=1, norm='batch', train_protocol=0):
    lgr = CLASS_NAME + "[convolutional_layer()]"

    if mode == '2D':
        return convolutional_layer_2D(x, num_of_iter, filters, kernel, stride, padding, activation, dilation, norm,
                                      train_protocol)
    else:
        return convolutional_layer_3D(x, num_of_iter, filters, kernel, stride, padding, activation, dilation, norm,
                                      train_protocol)


def convolutional_layer_3D(x, num_of_iter, filters, kernel, stride, padding, activation, dilation, norm,
                           train_protocol):
    for i in range(0, num_of_iter):
        x = Conv3D(filters, kernel, stride, padding=padding, dilation_rate=dilation)(x)
        if train_protocol == 0:
            x = Normalisation(x, norm)  # Original Model doesn't support batch normalization
            x = Activation(x, activation)  # Original Model -> PReLu
    # Following the steps from the CVPR paper, activation is applied once per each layer/block.
    if train_protocol > 0:
        x = BatchNormalization()(x)
        x = Activation(x, activation)

    return x


def convolutional_layer_2D(x, num_of_iter, filters, kernel, stride, padding, activation, dilation, norm, train_protocol):
    for i in range(0, num_of_iter):
        x = Conv2D(filters, kernel, stride, padding=padding, dilation_rate=dilation)(x)
        if train_protocol == 0:
            x = Normalisation(x, norm) # Original Model doesn't support batch normalization
            x = Activation(x, activation)  # Original Model -> PReLu
    if train_protocol > 0:
        x = Normalization(x, norm)
        x = Activation(x, activation)

    return x


def dimension_reduction_layer(x, filters, mode="3D"):
    lgr = CLASS_NAME + "[dimension_reduction_layer()]"

    if mode == "2D":
        return Conv2D(filters, (1, 1), padding='same')(x)
    else:
        return Conv3D(filters, (1, 1, 1), padding='same')(x)


def transpose_convolution(x, mode, filters, kernel, stride, padding, activation, dilation):
    if mode == "2D":
        return Conv2DTranspose(filters, kernel_size=kernel, strides=stride, padding=padding, dilation_rate=dilation,
                               activation=activation)(x)
    else:
        return Conv3DTranspose(filters, kernel_size=kernel, strides=stride, padding=padding, dilation_rate=dilation,
                               activation=activation)(x)
