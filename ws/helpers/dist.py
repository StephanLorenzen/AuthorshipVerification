import numpy as np

## Distance functions
def l1(d1, d2):
    l = min(d1.shape[0],d2.shape[0])
    return np.sum(np.abs(d1[:l]-d2[:l]))/l

