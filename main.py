# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

# Local imports
import helpers.data as avdata
from helpers.profiles import PROFILES
import helpers.util as util
import networks.n1 as n1
import networks.n2 as n2
import networks.n3 as n3

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    profile = PROFILES[dataset]

    model = n3.model(profile)

    print(model.summary())

    datagen = avdata.get_siamese_generator(datafile, dataset, ('char','word'))
    vdatagen = None
    if valdatafile is not None:
        vdatagen = avdata.get_siamese_generator(valdatafile, dataset, ('char','word'))
    
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1,
                            callbacks=[util.Checkpoint(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass

