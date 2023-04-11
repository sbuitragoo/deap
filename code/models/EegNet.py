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
from utils.utils import DepthwiseConv3D

## EEGNet model modified to be a 3D CNN.

def EEGNet_Full_3D(input_shape=(6,32,128), num_classes=1, WF=0.5):

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Temporal convolutional block using 3D convolutions
    x = Conv3D(int(16*WF), (input_shape[0], 1, 1), padding='same', strides=(1,2,2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    # Depthwise separable convolutional block using 3D depthwise separable convolutions
    irb1 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="relu")(x)
    irb1 = DepthwiseConv3D(kernel_size=(3, 3, 3), padding='valid')(irb1)
    irb1 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="linear")(irb1)
    irb1 = Add()([x, irb1])

    irb2 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="relu")(irb1)
    irb2 = DepthwiseConv3D(kernel_size=(3, 3, 3), padding='valid')(irb2)
    irb2 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="linear")(irb2)
    irb2 = Add()([irb1, irb2])

    irb3 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="relu")(irb2)
    irb3 = DepthwiseConv3D(kernel_size=(3, 3, 3), padding='valid')(irb3)
    irb3 = Conv3D(int(16*WF), (1, 1, 1), padding='same', activation="linear")(irb3)
    irb3 = Add()([irb2, irb3])

    x = AveragePooling3D(pool_size=(1,1,1))(irb3)

    x = Conv3D(16, (input_shape[0], 1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Flatten the output and pass it through a fully connected layer for classification
    x = Flatten()(x)
    x = Dense(64, 'relu')(x)
    x = Dense(32, 'relu')(x)
    x = Dense(8, 'relu')(x)
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