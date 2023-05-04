import numpy as np
np.set_printoptions(suppress=True)
def score(predict, label):
    ## inputs are two numpy array
    print(predict)
    print(label)
    quit()
    print(label - predict)
    quit()
    left = np.sqrt(np.square(label - predict).mean())
    right = np.square(np.abs(label - predict)).mean()
    return (left + right) / 2
