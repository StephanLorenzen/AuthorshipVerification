# External imports
import tensorflow as tf
import keras.layers as L
import keras.backend as K
import random

# Local imports
import util.data as avdata

CHAR_MAP_SIZE = 100
WORD_MAP_SIZE = 10000


def l1(A,B):
   return K.sum(K.abs(A-B),axis=1,keepdims=True)

def model():
    inshape = (None, )

    # Siamese part of network
    char_embd = L.Embedding(avdata.CHAR_MAP_SIZE, 5)
    word_embd = L.Embedding(avdata.WORD_MAP_SIZE, 8)
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

    reweight = L.Dense(k, activation='linear')

    inls  = []
    outls = []
    for name in ['known', 'unknown']:
        c_in = L.Input(shape=inshape, name=name+"_char_in", dtype='int32')
        w_in = L.Input(shape=inshape, name=name+"_word_in", dtype='int32')
        inls.append(c_in)
        inls.append(w_in)

        c_out = char_pool(char_conv(char_embd(c_in)))
        w_out = word_pool(word_conv(word_embd(w_in)))
    
        concat = L.Concatenate([c_out,w_out])
        
        outls.append(concat)


    output = Merge(mode=lambda x:l1(x[0],x[1]), output_shape=lambda in_shp: (in_shp[0][0],1))(outls)   

    model = Model(inputs=inls, outputs=outls)

    model.compile(
            optimizer=optimizer,
            loss='categorial_crossentropy',
            metrics=['accuracy']
            )

    return model


