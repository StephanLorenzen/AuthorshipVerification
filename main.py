# External imports
import tensorflow as tf
import numpy as np

# Local imports
import util.data as avdata
from util.profiles import PROFILES
import networks.n1 as n1

def train(datafile, valdatafile=None, dataset="MaCom"):
    inp,out = avdata.get_siamese_set(datafile, dataset)
    
    
    profile = PROFILES[dataset]

    model = n1.model(profile)

    if valdatafile is not None:
        vinp, vout = avdata.get_siamese_set(valdatafile, dataset)
        model.fit(inp, out, validation_data=(vinp, vout), epochs=10, verbose=1)
    else:
        model.fit(inp, out, epochs=10, verbose=1)

def test(datafile, dataset="MaCom"):
    pass



train("H","I")
