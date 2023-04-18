import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,data,window_size = 10):
        self.input = []
        self.output = []
        time = pd.DataFrame()
        time['TurbID'] = data['TurbID']
        time['Day'] = data['Day']
        time['Tmstamp'] = data['Tmstamp']
        train = data.drop(columns=['Day','Tmstamp']).groupby('TurbID')
        print(train)
        quit()


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        pass