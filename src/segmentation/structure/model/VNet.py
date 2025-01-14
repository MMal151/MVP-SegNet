from tensorflow.keras import models
from tensorflow.keras.layers import *

from src.segmentation.structure.layers.Convolutional import convolution_with_downsampling_layer, \
    convolution_with_upsampling_layer, convolutional_layer
from src.segmentation.structure.model.Model import Model

CLASS_NAME = "[Model/Vnet]"


class Vnet(Model):
    def __init__(self, input_shape, activation):
        super().__init__(input_shape)
        lgr = CLASS_NAME + "[init()]"

        self.num_encoder_blocks = [1, 2, 3, 3]
        self.num_decoder_blocks = [3, 3, 2, 1]
        self.use_transpose = True
        self.use_dlt_res_con = False
        self.sne_params = None
        self.use_sne = False
        self.add_res_cons = False

        # TO-DO: Make the params configurable.
        self.conv_params = {'kernel': 5, 'stride': 1, 'padding': 'same', 'activation': activation, 'dilation': 1,
                            'norm': 'batch', 'train_protocol': 0}
        self.down_params = {'kernel': 2, 'stride': 2, 'padding': 'same', 'activation': activation, 'dilation': 1,
                            'norm': 'batch', 'train_protocol': 0}
        self.up_params = {'kernel': 2, 'stride': 2, 'padding': 'same', 'activation': None, 'dilation': 1}

        self.print_info(lgr)

    def generate_model(self):
        img = Input(shape=self.input_shape)
        # Encoder

        rc_1, x = convolution_with_downsampling_layer(img, mode=self.conv_mode, filters=self.filters[0],
                                                      num_of_iter=self.num_encoder_blocks[0],
                                                      conv_params=self.conv_params,
                                                      down_params=self.down_params)
        rc_2, x = convolution_with_downsampling_layer(x, mode=self.conv_mode, filters=self.filters[1],
                                                      num_of_iter=self.num_encoder_blocks[1],
                                                      conv_params=self.conv_params,
                                                      down_params=self.down_params)
        rc_3, x = convolution_with_downsampling_layer(x, mode=self.conv_mode, filters=self.filters[2],
                                                      num_of_iter=self.num_encoder_blocks[2],
                                                      conv_params=self.conv_params,
                                                      down_params=self.down_params)
        rc_4, x = convolution_with_downsampling_layer(x, mode=self.conv_mode, filters=self.filters[3],
                                                      num_of_iter=self.num_encoder_blocks[3],
                                                      conv_params=self.conv_params,
                                                      down_params=self.down_params)

        # Bottleneck layer
        x = convolutional_layer(x, mode=self.conv_mode, filters=self.filters[4], num_of_iter=self.num_encoder_blocks[3],
                                **self.conv_params)

        x = convolution_with_upsampling_layer(x, rc_4, self.conv_mode,
                                              self.filters[4], self.num_decoder_blocks[0], self.use_transpose,
                                              self.conv_params,
                                              self.up_params)
        x = convolution_with_upsampling_layer(x, rc_3, self.conv_mode,
                                              self.filters[3], self.num_decoder_blocks[1], self.use_transpose,
                                              self.conv_params,
                                              self.up_params)
        x = convolution_with_upsampling_layer(x, rc_2, self.conv_mode,
                                              self.filters[2], self.num_decoder_blocks[2], self.use_transpose,
                                              self.conv_params,
                                              self.up_params)
        x = convolution_with_upsampling_layer(x, rc_1, self.conv_mode,
                                              self.filters[1], self.num_decoder_blocks[3], self.use_transpose,
                                              self.conv_params,
                                              self.up_params)

        if self.conv_mode == "2D":
            x = SpatialDropout2D(self.dropout)(x)
            output = Conv2D(self.output_classes, 1, activation='sigmoid')(
                x)  # Assumption: Output_classes = 1 - Activation will be changed for output classes > 1

        else:
            x = SpatialDropout3D(self.dropout)(x)
            output = Conv3D(self.output_classes, 1, activation='sigmoid')(x)

        # Update the output layer based on the number of classes
        out = models.Model(img, output, name='vnet')
        out.summary()
        return out
