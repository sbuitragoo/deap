import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv3D, MaxPooling2D, AveragePooling3D, Conv2D
from keras.layers import SeparableConv2D, DepthwiseConv2D, ReLU, Add
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K
# from utils.utils import DepthwiseConv3D

import tensorflow as tf

class DepthwiseConv3D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides=(1, 1, 1), padding='valid'):
        super(DepthwiseConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(
            shape=(*self.kernel_size, input_shape[-1], 1),
            initializer=tf.keras.initializers.glorot_uniform(),
            trainable=True,
            name='depthwise_kernel'
        )

    def call(self, inputs):
        outputs = tf.nn.depthwise_conv3d(
            inputs,
            self.depthwise_kernel,
            strides=(1, *self.strides, 1),
            padding=self.padding.upper()
        )
        return outputs

## EEGNet model modified to be a 3D CNN.

def EEGNet_Full_3D(input_shape, num_classes):

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Temporal convolutional block using 3D convolutions
    x = Conv3D(8, (1, 3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv3D(8, (input_shape[0], 1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Depthwise separable convolutional block using 3D depthwise separable convolutions
    x = DepthwiseConv3D(kernel_size=(input_shape[0], 1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (1, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = DepthwiseConv3D(kernel_size=(1, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Flatten the output and pass it through a fully connected layer for classification
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    return model

def EEGNet_3D(input_shape, num_classes):

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Temporal convolutional block using 2D convolutions
    x = Conv2D(8, (1, 64), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(8, (input_shape[0], 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)

    # Depthwise separable convolutional block using a combination of 2D and 3D convolutions
    x = DepthwiseConv2D((input_shape[0], 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv3D(16, (1, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)
    x = DepthwiseConv3D((1, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv3D(16, (1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)

    # Flatten the output and pass it through a fully connected layer for classification
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    return model



if __name__ == "__main__":
    pass