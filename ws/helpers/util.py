import os
from configparser import ConfigParser

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

def get_path(dataset, network):
    config = get_config()
    path = config['path_storage']+'ws/'+dataset+'/'+network+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

