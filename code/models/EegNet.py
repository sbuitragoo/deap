import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv3D, MaxPooling2D, AveragePooling3D
from keras.layers import SeparableConv2D, DepthwiseConv2D, ReLU, Add
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K
from utils.utils import DepthwiseConv3D, separable_conv3d

## EEGNet model modified to be a 3D CNN.


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', trials = 6):
    """ 
    Inputs:
        
    nb_classes      : int, number of classes to classify
    Chans, Samples  : number of channels and time points in the EEG data
    dropoutRate     : dropout fraction
    kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.      
    F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
    D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
    dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                        'or Dropout, passed as a string.')
    
    input1       = Input(shape = (trials, Chans, Samples))

    ###################################################################

    block1       = Conv3D(F1, kernel_size=(1, 3, 3), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, strides=(1,2,2))(input1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('relu')(block1)

    ########################Inverted Residual 1########################

    ir1          = Conv3D(F1, (1, 3, 32), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="relu", strides=(1,1,1))(block1)
    ir1       = DepthwiseConv3D((3,3,3), use_bias = False, 
                                depth_multiplier = D,
                                activation="relu",
                                depthwise_constraint = max_norm(1.))(ir1)
    ir1          = Conv3D(F1, (1, 3, 3), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="linear")(ir1)
    ir1          = Add()([block1, ir1])

    ########################Inverted Residual 2########################

    ir2          = Conv3D(F1, (1, 3, 32), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="relu", strides=(1,1,1))(ir1)
    ir2       = DepthwiseConv3D((3,3,3), use_bias = False, 
                                depth_multiplier = D,
                                activation="relu",
                                depthwise_constraint = max_norm(1.))(ir2)
    ir2          = Conv3D(F1, (1, kernLength), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="linear")(ir2)
    ir2          = Add()([ir1, ir2])

    ########################Inverted Residual 3########################

    ir3          = Conv3D(F1, (1, 3, 32), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="relu", strides=(1,1,1))(ir2)
    ir3       = DepthwiseConv3D((3,3,3), use_bias = False, 
                                depth_multiplier = D,
                                activation="relu",
                                depthwise_constraint = max_norm(1.))(ir3)
    ir3          = Conv3D(F1, (1, 3, 32), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False, activation="linear")(ir3)
    ir3          = Add()([ir2, ir3])

    ###################################################################

    block2       = Conv3D(F1, (1, 3, 32), padding = 'same',
                                input_shape = (Chans, Samples, 1),
                                use_bias = False)(ir3)
    block2       = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)

    do       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten(name = 'flatten')(do)
    
    dense        = Dense(nb_classes, name = 'dense', 
                        kernel_constraint = max_norm(norm_rate))(flatten)
    
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

if __name__ == "__main__":
    pass