import math

############### Combine functions
def cmin(sequence):
    return min([x[2] for x in sequence])

def cmax(sequence):
    return max([x[2] for x in sequence])

def uniform(sequence):
    return weighted(sequence, [1.0]*len(sequence))

def exponential(sequence, lt=0.0, ll=0.0):
    t0 = sequence[-1][0]
    times = [relative_months(x[0],t0) for x in sequence]
    lengths = [x[1]/10000.0 for x in sequence]
    weights = [math.exp(-lt*t+ll*l) for t,l in zip(times,lengths)]
    return weighted(sequence, weights)

def majority(sequence):
    return sum([(1 if x[2] > 0.5 else 0) for x in sequence]) / float(len(sequence))









############## Utilities
def relative_months(ts, t0):
    return (t0-ts) / (30.0*24.0*60.0*60.0)

def weighted(sequence, weights):
    assert len(sequence) == len(weights)

    return sum([w*x[2] for w,x in zip(weights,sequence)]) / sum(weights)

