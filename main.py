import argparse

import pandas as pd
import torch
from torch import cuda, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MyModel
import torch.utils.data as data
from data_utils import MyDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-ws', type=int, default=15)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    seed = args.seed
    window_size = args.ws
    batch_size = args.batch_size

    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    model = MyModel(args).to(device)

    df = pd.read_csv('data/clean_fill_data.csv')[:100000]

    train = df.drop(columns=['TurbID', 'Day', 'Tmstamp'])

    dataset = MyDataset(df, window_size=window_size)

    train_set, eval_set = data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
    eval_set, test_set = data.random_split(eval_set, [0.5, 0.5], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size)

    epoch = 20
    global_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fct = nn.MSELoss()
    for e in range(epoch):
        model.train()
        for i in tqdm(
                train_loader,
                # mininterval=200
        ):
            input_, output = i[0].to(device), i[1].to(device)
            attention_mask = torch.ones((input_.shape[0], 1, window_size - 1)).to(device)
            predict = model(input_, attention_mask)

            loss = loss_fct(predict, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 1000 == 0:
                print('loss: ', loss.item())
