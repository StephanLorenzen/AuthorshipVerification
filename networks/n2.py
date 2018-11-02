# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.optimizers as O
from keras.models import Model
import random

# Local imports
import helpers.data as avdata

def l1(A,B):
   return K.sum(K.abs(A-B),axis=1,keepdims=True)
def absdiff(A,B):
    return K.abs(A-B)

def model(dinfo):
    dinfo.channels(('char','word','pos'))

    # Siamese part of network
    char_embd = L.Embedding(dinfo.channel_size('char'), 5)
    word_embd = L.Embedding(dinfo.channel_size('word'), 8)
    pos_embd  = L.Embedding(dinfo.channel_size('pos'), 3)

    char_conv = L.Convolution1D(
        filters=400,
        kernel_size=8,
        strides=1,
        activation='relu',
        name='char_conv')
    word_conv = L.Convolution1D(
        filters=400,
        kernel_size=5,
        strides=1,
        activation='relu',
        name='word_conv')
    pos_conv = L.Convolution1D(
            filters=500,
            kernel_size=5,
            strides=1,
            activation='relu',
            name='pos_conv')
            
    char_pool = L.GlobalMaxPooling1D(name='char_pool')
    word_pool = L.GlobalMaxPooling1D(name='word_pool')
    pos_pool = L.GlobalMaxPooling1D(name='pos_pool')

    pos_lstm = L.LSTM(100, name='pos_lstm')

    reweight = L.Dense(1200, activation='linear', name='reweight')

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=(None,), name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=(None,), name=name+"_word_in", dtype='int32')
        p_in = L.Input(shape=(None,), name=name+"_pos_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)
        inls.append(p_in)

        c_out = char_pool(char_conv(char_embd(c_in)))
        w_out = word_pool(word_conv(word_embd(w_in)))
        #p_out = pos_pool(pos_conv(pos_embd(p_in)))
        p_out = pos_lstm(pos_embd(p_in))

        #concat = reweight(L.concatenate([c_out,w_out,p_out]))
        concat = L.concatenate([c_out,w_out,p_out])
        
        outls.append(concat)


    #dist = L.Lambda(lambda x:l1(x[0],x[1]), output_shape=lambda in_shp: (in_shp[0][0],1), name='distance')(outls)   
    dist = L.Lambda(lambda x:absdiff(x[0],x[1]), output_shape=lambda in_shp: in_shp, name='distance')(outls)
        
    output = L.Dense(2, activation='softmax', name='output')(dist)

    model = Model(inputs=inls, outputs=[output], name='n2')

    optimizer = O.Adam(lr=0.0005)

    model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

    return model


