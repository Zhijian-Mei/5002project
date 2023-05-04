import numpy as np


def score(predict, label):
    ## inputs are two numpy array
    left = np.sqrt(np.square(label - predict).mean())
    right = np.square(np.abs(label - predict)).mean()
    return (left + right) / 2
