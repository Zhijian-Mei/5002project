import numpy as np
np.set_printoptions(suppress=True)
def score_t(predict, label):
    ## inputs are two numpy array
    predict = predict / 1000
    label = label / 1000
    left = rmse(label - predict)
    right = mae(label - predict)
    return round((left + right) / 2,2)

def score_t_abnormal(predict, label):
    ## inputs are two numpy array
    predict = predict / 1000
    label = label / 1000
    diffs = np.empty_like(label)
    for i in range(label.shape[0]):
        if label[i] <= 0:
            diffs[i] = predict[i]
        elif np.isnan(label[i]):
            diffs[i] = 0
        else:
            diffs[i] = label[i] - predict[i]
    assert len(diffs) == 288
    left = rmse(diffs)
    right = mae(diffs)
    result = (left + right) / 2
    return left,right,result

def rmse(diffs):
    return np.sqrt(np.square(diffs).mean())

def mae(diffs):
    return np.abs(diffs).mean()