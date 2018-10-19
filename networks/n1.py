# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.optimizers as O
from keras.models import Model
import random

# Local imports
import util.data as avdata

CHAR_MAP_SIZE = 100
WORD_MAP_SIZE = 10000


def l1(A,B):
   return K.sum(K.abs(A-B),axis=1,keepdims=True)

def model(profile):
    # Siamese part of network
    char_embd = L.Embedding(profile["char_map_size"], 5)
    word_embd = L.Embedding(profile["word_map_size"], 8)
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

    reweight = L.Dense(400, activation='relu', name='reweight')
    dropout  = L.Dropout(0.3)

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=(10000,), name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=(3000,), name=name+"_word_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)

        c_out = char_pool(char_conv(char_embd(c_in)))
        w_out = word_pool(word_conv(word_embd(w_in)))
    
        concat = dropout(reweight(L.concatenate([c_out,w_out])))
        
        outls.append(concat)


    dist = L.Lambda(lambda x:l1(x[0],x[1]), output_shape=lambda in_shp: (in_shp[0][0],1), name='distance')(outls)

    output = L.Dense(2, activation='softmax', name='output')(dist)

    model = Model(inputs=inls, outputs=[output], name='n1')

    optimizer = O.Adam(lr=profile["lr"])

    model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

    return model


