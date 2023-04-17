from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,df):
        self.eval = eval
        self.text = list(df['text'])
        self.label = list(df['spans'])



    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = FloatTensor(self.label[idx])
        return text,label,len(text)