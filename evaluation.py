import numpy as np
np.set_printoptions(suppress=True)
def score_t(predict, label):
    ## inputs are two numpy array
    predict = predict / 1000
    label = label / 1000
    left = np.sqrt(np.square(label - predict).mean())
    right = np.square(np.abs(label - predict)).mean()
    return (left + right) / 2
