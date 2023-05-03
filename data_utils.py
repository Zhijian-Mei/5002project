import pandas as pd
from torch.utils.data import Dataset
from tqdm import trange


class MyDataset(Dataset):
    def __init__(self,data,ws=288):
        offset = 288
        self.input = []
        self.output = []
        self.id = []
        # time = pd.DataFrame()
        # time['TurbID'] = data['TurbID']
        # time['Day'] = data['Day']
        # time['Tmstamp'] = data['Tmstamp']
        train = list(data.groupby('TurbID'))
        print('loading data')
        for i in trange(len(train)):
            trub_id = i+1
            current = train[i][1].drop(columns=['TurbID']).reset_index(drop=True)
            for j in range(len(current)-offset-offset+1):
                input_ = current[j:j+offset].drop(columns=['Patv']).values
                output = current[j+offset:j+offset+offset]['Patv'].values / 100
                self.id.append(trub_id)
                self.input.append(input_)
                self.output.append(output)
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx],self.id[idx]