import pandas as pd
from torch.utils.data import Dataset
from tqdm import trange


class MyDataset(Dataset):
    def __init__(self,df,ws=288):
        offset = 288
        self.input = []
        self.output = []
        current = df.drop(columns=['TurbID']).reset_index(drop=True)
        for j in trange(len(current)):
            input_ = current[j:j + offset].drop(columns=['Patv']).values
            output = current[j + offset:j + offset + offset]['Patv'].values / 100
            self.input.append(input_)
            self.output.append(output)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]