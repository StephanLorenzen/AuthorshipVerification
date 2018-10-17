# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np
from io import open

# Local imports
import util.data as avdata
from util.profiles import PROFILES
import networks.n1 as n1

class MyCallback(Callback):
    def __init__(self, dataset):
        super(MyCallback, self).__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch, logs=None):
        wpath = "store/"+self.dataset+"/"+self.model.name+"-"+str(epoch)+".h5"
        self.model.save_weights(wpath)
        
        lpath = "store/"+self.dataset+"/"+self.model.name+"-log.txt"
        if epoch == 0:
            with open(lpath, 'w') as log:
                log.write("Epoch\tTrainAcc\tTrainLoss\tValAcc\tValLoss\n")
        
        if logs is not None:
            with open(lpath, 'a') as logfile:
                logfile.write(str(epoch)+"\t"+str(logs["acc"])+"\t"+str(logs["loss"])+"\t"+str(logs["val_acc"])+"\t"+str(logs["val_loss"])+"\n")

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    inp,out = avdata.get_siamese_set(datafile, dataset)
    
    
    profile = PROFILES[dataset]

    model = n1.model(profile)

    vdata = None
    if valdatafile is not None:
        vdata = avdata.get_siamese_set(valdatafile, dataset)
    
    model.fit(inp, out, validation_data=vdata, epochs=20, verbose=1, callbacks=[MyCallback(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass



train("B","H")
