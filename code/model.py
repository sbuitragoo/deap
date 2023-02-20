import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, AveragePooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L1L2

# Gaussian functional conectivity network taken from https://www.kaggle.com/code/dggarciam94/gfcnet-11-11-2022-version

class GFC(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.gammad = self.add_weight(name = 'gammad',
                                shape = (),
                                initializer = 'zeros',
                                trainable = True)
        super().build(batch_input_shape)

    def call(self, X): 
        X = tf.transpose(X, perm  = (0, 3, 1, 2)) #(N, F, C, T)
        R = tf.reduce_sum(tf.math.multiply(X, X), axis = -1, keepdims = True) #(N, F, C, 1)
        D  = R - 2*tf.matmul(X, X, transpose_b = True) + tf.transpose(R, perm = (0, 1, 3, 2)) #(N, F, C, C)

        ones = tf.ones_like(D[0,0,...]) #(C, C)
        mask_a = tf.linalg.band_part(ones, 0, -1) #Upper triangular matrix of 0s and 1s (C, C)
        mask_b = tf.linalg.band_part(ones, 0, 0)  #Diagonal matrix of 0s and 1s (C, C)
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) #Make a bool mask (C, C)
        triu = tf.expand_dims(tf.boolean_mask(D, mask, axis = 2), axis = -1) #(N, F, C*(C-1)/2, 1)
        sigma = tfp.stats.percentile(tf.math.sqrt(triu), 50, axis = 2, keepdims = True) #(N, F, 1, 1)

        A = tf.math.exp(-1/(2*tf.pow(10., self.gammad)*tf.math.square(sigma))*D) #(N, F, C, C)
        A.set_shape(D.shape)
        return A

    def compute_output_shape(self, batch_input_shape):
        N, C, T, F = batch_input_shape.as_list()
        return tf.TensorShape([N, F, C, C])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class get_triu(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, X): 
        N, F, C, C = X.shape
        ones = tf.ones_like(X[0,0,...]) #(C, C)
        mask_a = tf.linalg.band_part(ones, 0, -1) #Upper triangular matrix of 0s and 1s (C, C)
        mask_b = tf.linalg.band_part(ones, 0, 0)  #Diagonal matrix of 0s and 1s (C, C)
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) #Make a bool mask (C, C)
        triu = tf.expand_dims(tf.boolean_mask(X, mask, axis = 2), axis = -1) #(N, F, C*(C-1)/2, 1)

        triu.set_shape([N,F,int(C*(C-1)/2),1])
        return triu

    def compute_output_shape(self, batch_input_shape):
        N, F, C, C = batch_input_shape.as_list()
        return tf.TensorShape([N, F, int(C*(C-1)/2),1])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
    
    
def GFC_triu_net_avg(nb_classes: int,
          Chans: int,
          Samples: int,
          l1: int = 0, 
          l2: int = 0, 
          dropoutRate: float = 0.5,
          filters: int = 1, 
          maxnorm: float = 2.0,
          maxnorm_last_layer: float = 0.5,
          kernel_time_1: int = 20,
          strid_filter_time_1: int = 1,
          bias_spatial: bool = False) -> Model:


    input_main   = Input((Chans, Samples, 1),name='Input')                    
    
    block        = Conv2D(filters,(1,kernel_time_1),strides=(1,strid_filter_time_1),
                            use_bias=bias_spatial,
                            kernel_constraint = max_norm(maxnorm, axis=(0,1,2))
                            )(input_main)
    
    block        = BatchNormalization(epsilon=1e-05, momentum=0.1)(block)

    block        = Activation('elu')(block)      
    
    block        = GFC()(block)

    block        = get_triu()(block)

    block        = AveragePooling2D(pool_size=(block.shape[1],1),strides=(1,1))(block)
    
    block        = BatchNormalization(epsilon=1e-05, momentum=0.1)(block)

    block        = Activation('elu')(block) 
    
    block        = Flatten()(block)    

    block        = Dropout(dropoutRate)(block) 

    block        = Dense(nb_classes, kernel_regularizer=L1L2(l1=l1,l2=l2),name='logits',
                              kernel_constraint = max_norm(maxnorm_last_layer)
                              )(block)

    softmax      = Activation('softmax',name='output')(block)
    
    return Model(inputs=input_main, outputs=softmax)