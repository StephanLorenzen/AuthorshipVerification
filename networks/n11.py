# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.optimizers as O
from keras.models import Model
import random
import os

# Local imports
import helpers.data as avdata

def absdiff(A,B):
    return K.abs(A-B)

def model(dinfo):
    dinfo.channels(('char','word','pos'))
    dinfo.batch_size(8)

    # Siamese part of network
    char_embd = L.Embedding(dinfo.channel_size('char'), 5, name='char_embedding')
    word_embd = L.Embedding(dinfo.channel_size('word'), 8, name='word_embedding')
    pos_embd  = L.Embedding(dinfo.channel_size('pos'), 2, name='pos_embedding')

    char_conv4 = L.Convolution1D(
        filters=500,
        kernel_size=4,
        strides=1,
        activation='relu',
        name='char_conv4')
    word_conv3 = L.Convolution1D(
        filters=500,
        kernel_size=3,
        strides=1,
        activation='relu',
        name='word_conv3')
    pos_conv3 = L.Convolution1D(
            filters=500,
            kernel_size=3,
            strides=1,
            activation='relu',
            name='pos_conv3')
    pos_conv6 = L.Convolution1D(
            filters=500,
            kernel_size=3,
            strides=1,
            activation='relu',
            name='pos_conv6')
            
    char_pool4 = L.GlobalMaxPooling1D(name='char_pool4')
    word_pool3 = L.GlobalMaxPooling1D(name='word_pool3')
    pos_pool3 = L.GlobalMaxPooling1D(name='pos_pool3')
    pos_pool6 = L.GlobalMaxPooling1D(name='pos_pool6')

    char_dense4 = L.Dense(100, activation='relu')
    word_dense3 = L.Dense(100, activation='relu')
    pos_dense3 = L.Dense(100, activation='relu')
    pos_dense6 = L.Dense(100, activation='relu')

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=(None,), name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=(None,), name=name+"_word_in", dtype='int32')
        p_in = L.Input(shape=(None,), name=name+"_pos_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)
        inls.append(p_in)
        
        c_out4 = char_dense4(char_pool4(char_conv4(char_embd(c_in))))
        w_out3 = word_dense3(word_pool3(word_conv3(word_embd(w_in))))
        p_out3 = pos_dense3(pos_pool3(pos_conv3(pos_embd(p_in))))
        p_out6 = pos_dense6(pos_pool6(pos_conv6(pos_embd(p_in))))

        concat = L.concatenate([c_out4,w_out3,p_out3,p_out6])
        
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
