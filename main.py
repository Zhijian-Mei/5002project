import argparse

import pandas as pd
import torch
from torch import cuda
from model import MyModel

from data_utils import MyDataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-hidden_size', type=int, default=32)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')
    model = MyModel(args).to(device)

    data = pd.read_csv('data/clean_fill_data.csv')
    time = pd.DataFrame()
    time['TurbID'] = data['TurbID']
    time['Day'] = data['Day']
    time['Tmstamp'] = data['Tmstamp']
    train = data.drop(columns=['TurbID','Day','Tmstamp'])
    # input = torch.tensor(train[1:16].values).to(torch.float32).reshape((1,15,10)).to(device) # batch_size x sequence length x embedding size
    # attention_mask = torch.ones((1,1,15)).to(device)  # batch_size x 1 x sequence length
    # target = torch.tensor(train[11:12].values).to(device)
    # output = model(input,attention_mask)
    # print(output.shape)

    dataset = MyDataset(data,window_size=15)
