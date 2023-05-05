import numpy as np
np.set_printoptions(suppress=True)
def score(predict, label):
    ## inputs are two numpy array
    predict = predict / 1000
    label = label / 1000
    print(np.square(label - predict))
    quit()
    left = np.sqrt(np.square(label - predict).mean())
    right = np.square(np.abs(label - predict)).mean()
    return (left + right) / 2
