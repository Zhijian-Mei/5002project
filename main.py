import argparse

import numpy as np
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
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-gpu', type=str, default='7')
    parser.add_argument('-hidden_size', type=int, default=512)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-ws', type=int, default=288)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    seed = args.seed
    batch_size = args.batch_size
    ws = args.ws
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    model = MyModel(args).to(device)

    df = pd.read_csv('data/clean_fill_data.csv')[:10000]

    train = df.drop(columns=['TurbID', 'Day', 'Tmstamp'])

    dataset = MyDataset(df,ws=ws)

    train_set, eval_set = data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
    eval_set, test_set = data.random_split(eval_set, [0.5, 0.5], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size,shuffle=False)

    epoch = 20
    global_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    loss_fct = nn.MSELoss()
    best_eval_loss = np.inf
    for e in range(epoch):
        epoch_loss = 0
        model.train()
        for i in tqdm(
                train_loader,
                # mininterval=200
        ):
            input_, output = i[0].to(device), i[1].to(device)
            print(input_.shape)
            print(output.shape)
            quit()
            attention_mask = torch.ones((input_.shape[0], 1, ws)).to(device)
            predict = model(input_, attention_mask)
            loss = loss_fct(predict, output)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_loss += input_.shape[0] * loss.item()

        print(f'total train loss at epoch {e}: {epoch_loss}')

        model.eval()
        eval_loss = 0
        for i in tqdm(
                eval_loader,
                # mininterval=200
        ):
            input_, output = i[0].to(device), i[1].to(device)
            attention_mask = torch.ones((input_.shape[0], 1, 1)).to(device)
            predict = model(input_, attention_mask)

            loss = loss_fct(predict, output)
            eval_loss += input_.shape[0] * loss.item()

        print(f'total eval loss at epoch {e}: {eval_loss}')
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({'model': model.state_dict()},f'checkpoint/best_epoch{e}_loss_{round(best_eval_loss, 3)}.pt')
            print('saving better checkpoint')




