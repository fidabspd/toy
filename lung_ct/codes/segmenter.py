import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


class Encdoer(Layer):
    
    def __init__(self, n_filters, kernel_size=3, activation='relu', padding='same', **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = [
                Conv2D(n_filter, kernel_size=kernel_size, activation=activation, padding=padding)
                for n_filter in n_filters
        ]
        self.pool_layers = [
                MaxPool2D(pool_size=2, padding='same')
                for _ in n_filters
        ]
        self.fc = Dense(n_filters[-1], activation=activation)

    @tf.function
    def call(self, inputs):
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            inputs = conv_layer(inputs)
            inputs = pool_layer(inputs)
        outputs = self.fc(inputs)
        return outputs
        

class Decdoer(Layer):
    
    def __init__(self, n_filters, kernel_size=3, size=2, activation='sigmoid', padding='same', **kwargs):

        assert n_filters[-1] == 1, f'Last kernel_size must be 1. Now: {n_filters}'

        super().__init__(**kwargs)
        self.up_sample_layers = [
                UpSampling2D(size=size)
                for _ in n_filters
        ]
        self.conv_layers = [
                Conv2D(n_filter, kernel_size=kernel_size, activation=activation, padding=padding)
                for n_filter in n_filters
        ]

    @tf.function
    def call(self, encd):
        decd = encd
        for up_sample_layer, conv_layer in zip(self.up_sample_layers, self.conv_layers):
            decd = up_sample_layer(decd)
            decd = conv_layer(decd)
        return decd


class Segmenter(Model):
    
    def __init__(self, encdoer_n_filters, decoder_n_filters,
                 encoder_kernel_size=3, decoder_kernel_size=3,
                 decoder_up_sample_size=2, 
                 encoder_activation='relu', decoder_activation='sigmoid', 
                 padding='same',
                 encoder_name='None', decoder_name='None', **kwargs):

        super().__init__(**kwargs)
        self.encoder = Encdoer(encdoer_n_filters, kernel_size=encoder_kernel_size,
                               activation=encoder_activation, padding=padding,
                               name=encoder_name)
        self.decoder = Decdoer(decoder_n_filters, kernel_size=decoder_kernel_size,
                               size=decoder_up_sample_size, activation=decoder_activation,
                               padding=padding, name=decoder_name)

    @tf.function
    def call(self, inputs):
        encd = self.encoder(inputs)
        decd = self.decoder(encd)
        return decd