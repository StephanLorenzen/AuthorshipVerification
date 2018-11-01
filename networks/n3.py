# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.optimizers as O
from keras.models import Model
import random

# Local imports
import helpers.data as avdata

def absdiff(A,B):
    return K.abs(A-B)

def model(profile, datagen):
    # Siamese part of network
    char_embd = L.Embedding(datagen.channel_size('char'), 5)
    word_embd = L.Embedding(datagen.channel_size('word'), 8)
    #TODO POS-input

    char_conv = L.Convolution1D(
        filters=500,
        kernel_size=8,
        strides=1,
        activation='relu',
        name='char_conv')
    word_conv = L.Convolution1D(
        filters=500,
        kernel_size=5,
        strides=1,
        activation='relu',
        name='word_conv')

    char_pool = L.GlobalMaxPooling1D(name='char_pool')
    word_pool = L.GlobalMaxPooling1D(name='word_pool')

    #reweight = L.Dense(400, activation='relu', name='reweight')
    #dropout  = L.Dropout(0.3)

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=(None,), name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=(None,), name=name+"_word_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)

        c_out = char_pool(char_conv(char_embd(c_in)))
        w_out = word_pool(word_conv(word_embd(w_in)))
    
        #concat = dropout(reweight(L.concatenate([c_out,w_out])))
        concat = c_out
        
        outls.append(concat)

    dist = L.Lambda(lambda x:absdiff(x[0],x[1]), output_shape=lambda in_shp: in_shp, name='distance')(outls)

    output = L.Dense(2, activation='softmax', name='output')(dist)

    model = Model(inputs=inls, outputs=[output], name='n1')

    optimizer = O.Adam(lr=profile["lr"])

    model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

    return model


