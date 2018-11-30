import re
import os
from configparser import ConfigParser

from keras.callbacks import Callback

class Checkpoint(Callback):
    def __init__(self, dataset, epochoffset=0):
        super(Checkpoint, self).__init__()
        self.dataset = dataset
        self.offset = epochoffset
    
    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + self.offset + 1
        
        config = get_config()
       
        path = get_path(self.dataset,self.model.name)
        wpath = path+str(epoch)+'.h5'
        lpath = path+"log.txt"
        
        self.model.save_weights(wpath)
        if epoch == 1:
            with open(lpath, 'w', encoding="utf8") as log:
                log.write("Epoch\tTrainAcc\tTrainLoss\tValAcc\tValLoss\n")
        
        if logs is not None:
            with open(lpath, 'a', encoding="utf8") as logfile:
                logfile.write(str(epoch)+"\t"+str(logs["acc"])+"\t"+str(logs["loss"])+
                        "\t"+str(logs["val_acc"])+"\t"+str(logs["val_loss"])+"\n")

def clean(txt):
    txt = re.sub(r'\$NL\$', '\n', txt)
    txt = re.sub(r'\$SC\$', ';', txt)
    return txt

_CONFIG = None
def get_config():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    _CONFIG = ConfigParser()
    _CONFIG.optionxform = str
    _CONFIG.read('config.ini')
    _CONFIG = _CONFIG['Config']
    return _CONFIG

def get_path(dataset,network):
    config = get_config()
    path = config['path_storage']+'sim/'+dataset+'/'+network+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_data_path(dataset):
    config = get_config()
    path = config['path_data']+dataset+'/'
    propath = path+'processed/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

