# External imports
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np
from io import open

# Local imports
import util.data as avdata
from util.profiles import PROFILES
import networks.n1 as n1
import networks.n2 as n2

class MyCallback(Callback):
    def __init__(self, dataset):
        super(MyCallback, self).__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch, logs=None):
        wpath = "store/"+self.dataset+"/"+self.model.name+"-"+str(epoch)+".h5"
        self.model.save_weights(wpath)
        
        lpath = "store/"+self.dataset+"/"+self.model.name+"-log.txt"
        if epoch == 0:
            with open(lpath, 'w', encoding="utf8") as log:
                log.write(("Epoch\tTrainAcc\tTrainLoss\tValAcc\tValLoss\n"))
        
        if logs is not None:
            with open(lpath, 'a', encoding="utf8") as logfile:
                logfile.write((str(epoch)+"\t"+str(logs["acc"])+"\t"+str(logs["loss"])+"\t"+str(logs["val_acc"])+"\t"+str(logs["val_loss"])+"\n"))

def train(datafile, valdatafile=None, dataset="MaCom", network='n1'):
    profile = PROFILES[dataset]

    model = n1.model(profile)

    print(model.summary())

    datagen = avdata.get_siamese_generator(datafile, dataset, ('char','word'))
    vdatagen = None
    if valdatafile is not None:
        vdatagen = avdata.get_siamese_generator(valdatafile, dataset, ('char','word'))
    
    model.fit_generator(generator=datagen, validation_data=vdatagen, epochs=40, verbose=1, callbacks=[MyCallback(dataset)])
    
def test(datafile, dataset="MaCom"):
    pass



#train("test","train", "PAN13")
train("B","C")
