# External imports
import tensorflow as tf

# Local imports
import util.data as avdata
import networks.n1 as n1

def train(datafile):
    dataset,labels = avdata.get_siamese_set("I")

    known_char_in = [x[0][0] for x in dataset]
    known_word_in = [x[0][1] for x in dataset]
    unknown_char_in = [x[1][0] for x in dataset]
    unknown_word_in = [x[1][1] for x in dataset]

    model = n1.model()
    print(model.summary())

    model.fit({'known_char_in':known_char_in,'known_word_in':known_word_in,'unknown_char_in':unknown_char_in,'unknown_word_in':unknown_word_in}, labels, epochs=10, verbose=1)

def test(datafile):
    pass




