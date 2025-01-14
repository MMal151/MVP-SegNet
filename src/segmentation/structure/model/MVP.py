from tensorflow.keras import models
from tensorflow.keras.layers import *

from src.segmentation.structure.layers.Connections import dilated_skip_connection, dilated_skip_res_connection, \
    dilated_sne_skip_connection, dilated_sne_skip_res_connection, sne_skip_connection, sne_skip_res_connection
from src.segmentation.structure.layers.Convolutional import convolution_with_downsampling_layer, \
    convolution_with_upsampling_layer, convolutional_layer
from src.segmentation.structure.layers.SnE import sne_down_layer, sne_up_layer
from src.segmentation.structure.model.Model import Model

CLASS_NAME = "[Model/MVP]"


class MVP(Model):
    def __init__(self, input_shape, activation):
        super().__init__(input_shape)
        lgr = CLASS_NAME + "[init()]"

        cfg = self.cfg["mvp"]  # Separating out V-Net specific configurations

        num_encoder_blocks = cfg["encoder_blocks"].split(",")

        if len(num_encoder_blocks) >= 4:
            self.num_encoder_blocks = [int(i) for i in num_encoder_blocks]
        else:
            print(f"{lgr}[WARNING]: Invalid number of encoder blocks configured. The total number of blocks should"
                  f"be 4. Default value of 1,2,3,3 will be used. ")
            self.num_encoder_blocks = [1, 2, 3, 3]

        num_decoder_blocks = cfg["decoder_blocks"].split(",")
        if len(num_decoder_blocks) >= 3:
            self.num_decoder_blocks = [int(i) for i in num_decoder_blocks]
        else:
            print(f"{lgr}[WARNING]: Invalid number of decoder blocks configured. The total number of blocks should"
                  f"be at least 3. Default value of 3,3,2 will be used. ")
            self.num_decoder_blocks = [3, 3, 2, 1]

        self.use_transpose = cfg["use_transpose"]
        self.use_dlt_res_con = cfg["dilated_res_con"]["use_dltd_res_con"]
        self.dilation_rates = [1, 2, 4, 8]

        if self.use_dlt_res_con:
            dilation_rates = cfg["dilated_res_con"]["dilation_rates"].split(",")
            if len(dilation_rates) >= 4:
                self.dilation_rates = [int(i) for i in dilation_rates]
            else:
                print(f"{lgr}[WARNING]: Invalid dilatation rates configured. Default value: [1, 2, 4, 8] will be used.")

        self.sne_params = None
        self.use_sne = cfg["squeeze_excitation"]["use_sne"]
        if self.use_sne:
            print(f"{lgr}[INFO]: Using Squeeze and Excitation blocks for residual connection in up-sampling layer.")
            self.sne_params = {'ratio': cfg["squeeze_excitation"]["ratio"],
                               'use_res': cfg["squeeze_excitation"]["use_res_con"]}
            self.sne_mode = cfg["squeeze_excitation"][
                "mode"]  # if 'res', squeeze and excitation block will be applied in residual connection.
            # If 'ed', squeeze and excitation block will be applied in each encoder-decoder layer, before down or up sampling layer.
            # If 'enc', squeeze and excitation block will be applied to encoder block only and similarly if 'dec', sne block will be applied to decoder only.

        self.add_res_cons = cfg["add_both_res_con"]

        # TO-DO: Make the params configurable.
        self.conv_params = {'kernel': 5, 'stride': 1, 'padding': 'same', 'activation': activation, 'dilation': 1,
                            'norm': 'batch', 'train_protocol': 0}
        self.down_params = {'kernel': 2, 'stride': 2, 'padding': 'same', 'activation': activation, 'dilation': 1,
                            'norm': 'batch', 'train_protocol': 0}
        self.up_params = {'kernel': 2, 'stride': 2, 'padding': 'same', 'activation': None, 'dilation': 1}
        self.dilated_params = {'kernel': 3, 'stride': 1, 'padding': 'same', 'activation': activation,
                               'dilation': self.dilation_rates, 'norm': 'batch', 'train_protocol': 0}

        self.print_info(lgr)

    def get_connection(self, x, filters=None):
        if self.use_dlt_res_con and not (
                self.use_sne or (self.use_sne and self.sne_mode != "res")) and not self.add_res_cons:
            return dilated_skip_connection(x, self.conv_mode, filters, self.dilated_params)
        elif self.use_dlt_res_con and not (
                self.use_sne or (self.use_sne and self.sne_mode != "res")) and self.add_res_cons:
            return dilated_skip_res_connection(x, self.conv_mode, filters, self.dilated_params)
        elif self.use_dlt_res_con and (self.use_sne and self.sne_mode == 'res') and not self.add_res_cons:
            return dilated_sne_skip_connection(x, self.conv_mode, filters, self.dilated_params, self.sne_params)
        elif self.use_dlt_res_con and (self.use_sne and self.sne_mode == 'res') and self.add_res_cons:
            return dilated_sne_skip_res_connection(x, self.conv_mode, filters, self.dilated_params, self.sne_params)
        elif not self.use_dlt_res_con and (self.use_sne and self.sne_mode == 'res') and not self.add_res_cons:
            return sne_skip_connection(x, self.conv_mode, self.sne_params)
        elif not self.use_dlt_res_con and (self.use_sne and self.sne_mode == 'res') and self.add_res_cons:
            return sne_skip_res_connection(x, self.conv_mode, self.sne_params)
        return x

    def generate_model(self):
        img = Input(shape=self.input_shape)
        # Encoder

        if self.use_sne and (self.sne_mode == 'enc' or self.sne_mode == 'ed'):
            rc_1, x = sne_down_layer(img, mode=self.conv_mode, filters=self.filters[0],
                                     num_conv_blocks=self.num_encoder_blocks[0], conv_params=self.conv_params,
                                     sne_params=self.sne_params,
                                     down_params=self.down_params)
            rc_2, x = sne_down_layer(x, mode=self.conv_mode, filters=self.filters[1],
                                     num_conv_blocks=self.num_encoder_blocks[1],
                                     conv_params=self.conv_params, sne_params=self.sne_params,
                                     down_params=self.down_params)
            rc_3, x = sne_down_layer(x, mode=self.conv_mode, filters=self.filters[2],
                                     num_conv_blocks=self.num_encoder_blocks[2],
                                     conv_params=self.conv_params, sne_params=self.sne_params,
                                     down_params=self.down_params)
            rc_4, x = sne_down_layer(x, mode=self.conv_mode, filters=self.filters[3],
                                     num_conv_blocks=self.num_encoder_blocks[3],
                                     conv_params=self.conv_params, sne_params=self.sne_params,
                                     down_params=self.down_params)
        else:
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

        if self.use_sne and (self.sne_mode == 'dec' or self.sne_mode == 'ed'):
            self.dilated_params["dilation"] = self.dilation_rates[0:1]
            x = sne_up_layer(x, self.get_connection(rc_4, self.filters[3]), mode=self.conv_mode,
                             filters=self.filters[4], num_conv_blocks=self.num_decoder_blocks[0],
                             use_transpose=self.use_transpose, sne_params=self.sne_params,
                             conv_params=self.conv_params, up_params=self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates[0:2]
            x = sne_up_layer(x, self.get_connection(rc_3, self.filters[2]), mode=self.conv_mode,
                             filters=self.filters[3], num_conv_blocks=self.num_decoder_blocks[1],
                             use_transpose=self.use_transpose, sne_params=self.sne_params,
                             conv_params=self.conv_params, up_params=self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates[0:3]
            x = sne_up_layer(x, self.get_connection(rc_2, self.filters[1]), mode=self.conv_mode,
                             filters=self.filters[2], num_conv_blocks=self.num_decoder_blocks[2],
                             use_transpose=self.use_transpose, sne_params=self.sne_params,
                             conv_params=self.conv_params, up_params=self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates
            x = sne_up_layer(x, self.get_connection(rc_1, self.filters[0]), mode=self.conv_mode,
                             filters=self.filters[1], num_conv_blocks=self.num_decoder_blocks[3],
                             use_transpose=self.use_transpose, sne_params=self.sne_params,
                             conv_params=self.conv_params, up_params=self.up_params)
        else:
            self.dilated_params["dilation"] = self.dilation_rates[0:1]
            x = convolution_with_upsampling_layer(x, self.get_connection(rc_4, self.filters[3]), self.conv_mode,
                                                  self.filters[4], self.num_decoder_blocks[0], self.use_transpose,
                                                  self.conv_params,
                                                  self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates[0:2]
            x = convolution_with_upsampling_layer(x, self.get_connection(rc_3, self.filters[2]), self.conv_mode,
                                                  self.filters[3], self.num_decoder_blocks[1], self.use_transpose,
                                                  self.conv_params,
                                                  self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates[0:3]
            x = convolution_with_upsampling_layer(x, self.get_connection(rc_2, self.filters[1]), self.conv_mode,
                                                  self.filters[2], self.num_decoder_blocks[2], self.use_transpose,
                                                  self.conv_params,
                                                  self.up_params)
            self.dilated_params["dilation"] = self.dilation_rates
            x = convolution_with_upsampling_layer(x, self.get_connection(rc_1, self.filters[0]), self.conv_mode,
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
