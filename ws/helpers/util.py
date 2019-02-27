import os
from configparser import ConfigParser
import numpy as np

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

def get_path(datarepo, dataset, network):
    config = get_config()
    path = config['path_storage']+'ws/'+datarepo+'/'+network+'-'+dataset+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_data_path(datarepo):
    config = get_config()
    path = config['path_data']+datarepo+'/processed/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Interpolation

# DELTA value 
DELTA = 0.05

def _eq(p,q):
    (px,py) = p
    (qx,qy) = q
    a = (py-qy)/(px-qx)
    b = py - a*px
    return (a,b)
def _getEqs(data):
    res = []
    for ds in data:
        eqs = []
        for i in range(1,len(ds)):
            eq = _eq(ds[i-1],ds[i])
            eqs.append((ds[i][0],eq))
        res.append(eqs)
    return res
def _interpolate(data):
    eqs = _getEqs(data)
    res = []
    for seq in eqs:
        t = 0
        i  = 0
        nseq = []
        while i < len(seq):
            (et, (a,b)) = seq[i]
            if t > et:
                i += 1
            else:
                nseq.append(a*t+b)
                t += DELTA
        res.append(np.array(nseq))
    return res


