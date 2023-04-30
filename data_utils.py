import pandas as pd
from torch.utils.data import Dataset
from tqdm import trange


class MyDataset(Dataset):
    def __init__(self,data,window_size = 10):
        self.input = []
        self.output = []
        self.id = []
        time = pd.DataFrame()
        time['TurbID'] = data['TurbID']
        time['Day'] = data['Day']
        time['Tmstamp'] = data['Tmstamp']
        train = list(data.drop(columns=['Day','Tmstamp']).groupby('TurbID'))
        for i in trange(len(train)):
            trub_id = i+1
            current = train[i][1].drop(columns=['TurbID']).reset_index(drop=True)
            for j in range(len(current)-289):
                input_ = current[j:j+1].values
                output = current[j+1:j+289].values
                print(output)
                quit()
                self.id.append(trub_id)
                self.input.append(input_)
                self.output.append(output)
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx],self.id[idx]