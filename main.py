# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

# Local imports
import helpers.data as avdata
from helpers.profiles import PROFILES
import helpers.util as util
from networks import n1,n2,n3,n4

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    profile = PROFILES[dataset]

    model = n4.model(profile)

    print(model.summary())

    datagen = avdata.get_siamese_generator(datafile, dataset, ('char','word'))
    vdatagen = None
    if valdatafile is not None:
        vdatagen = avdata.get_siamese_generator(valdatafile, dataset, ('char','word'))
    
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass

