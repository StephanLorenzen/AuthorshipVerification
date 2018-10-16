# External imports
import tensorflow as tf
import numpy as np

# Local imports
import util.data as avdata
from util.profiles import PROFILES
import networks.n1 as n1

def train(datafile, dataset="MaCom"):
    data,labels = avdata.get_siamese_set(datafile,dataset)
    profile = PROFILES[dataset]

    known_char_in = np.array([x[0][0] for x in data])
    known_word_in = np.array([x[0][1] for x in data])
    unknown_char_in = np.array([x[1][0] for x in data])
    unknown_word_in = np.array([x[1][1] for x in data])
    labels = np.array(labels)

    model = n1.model(profile)

    model.fit({'known_char_in':known_char_in,'known_word_in':known_word_in,'unknown_char_in':unknown_char_in,'unknown_word_in':unknown_word_in}, {'output':labels}, epochs=5, verbose=1)

def test(datafile, dataset="MaCom"):
    pass




