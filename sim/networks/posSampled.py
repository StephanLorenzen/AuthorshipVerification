# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.optimizers as O
from keras.models import Model
import random
import os

def l1(A,B):
   return K.sum(K.abs(A-B),axis=1,keepdims=True)
def absdiff(A,B):
    return K.abs(A-B)

def model(dinfo):
    dinfo.channels(('char','word','pos-sampled'))
    dinfo.batch_size(8)

    # Siamese part of network
    char_embd = L.Embedding(dinfo.channel_size('char'), 5, name='char_embedding')
    word_embd = L.Embedding(dinfo.channel_size('word'), 8, name='word_embedding')
    pos_embd  = L.Embedding(dinfo.channel_size('pos'), 2, name='pos_embedding')

    char_conv = L.Convolution1D(
        filters=500,
        kernel_size=4,
        strides=1,
        activation='relu',
        name='char_conv')
    word_conv = L.Convolution1D(
        filters=500,
        kernel_size=3,
        strides=1,
        activation='relu',
        name='word_conv')
            
    char_pool = L.GlobalMaxPooling1D(name='char_pool')
    word_pool = L.GlobalMaxPooling1D(name='word_pool')

    pos_lstm = L.LSTM(100, name='pos_lstm')

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=(None,), name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=(None,), name=name+"_word_in", dtype='int32')
        p_in = L.Input(shape=(None,), name=name+"_pos-sampled_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)
        inls.append(p_in)

        c_out = char_pool(char_conv(char_embd(c_in)))
        w_out = word_pool(word_conv(word_embd(w_in)))
        p_out = pos_lstm(pos_embd(p_in))

        concat = L.concatenate([c_out,w_out,p_out])
        
        outls.append(concat)


    dist = L.Lambda(lambda x:absdiff(x[0],x[1]), output_shape=lambda in_shp: in_shp, name='distance')(outls)
        
    output = L.Dense(2, activation='softmax', name='output')(dist)

    mname = os.path.basename(__file__)[:-3]
    model = Model(inputs=inls, outputs=[output], name=mname)

    optimizer = O.Adam(lr=0.0005)

    model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

    return model
