import re
from keras.callbacks import Callback

class Checkpoint(Callback):
    def __init__(self, dataset):
        super(Checkpoint, self).__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch, logs=None):
        wpath = "store/"+self.dataset+"/"+self.model.name+"-"+str(epoch)+".h5"
        self.model.save_weights(wpath)
        
        lpath = "store/"+self.dataset+"/"+self.model.name+"-log.txt"
        if epoch == 0:
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