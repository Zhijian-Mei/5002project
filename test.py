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
from evaluation import score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-ws', type=int, default=288)
    parser.add_argument('-debug', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-epoch', type=int, default=200)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    args = get_args()
    print(args)
    gpu = args.gpu
    seed = args.seed
    batch_size = args.batch_size
    ws = args.ws
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    if args.debug:
        df = pd.read_csv('data/clean_fill_data.csv')[:10000]
    else:
        df = pd.read_csv('data/clean_fill_data.csv')
    subset = ['TurbID', 'Wspd', 'Wdir', 'Patv']
    df = df[subset]

    checkpoint = torch.load('checkpoint/best_epoch5_loss_inf.pt')
    model = MyModel(args, len(subset) - 2, device)
    model.load_state_dict(checkpoint['model'])

    train = df

    dataset = MyDataset(train, ws=ws)
    print('number of samples: ', len(dataset))
    train_set, eval_set, = data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    epoch = args.epoch
    loss_fct = nn.MSELoss()
    best_eval_loss = np.inf
    count = 0
    for e in range(epoch):
        epoch_loss = 0
        model.train()
        for i in tqdm(
                eval_loader,
                mininterval=200
        ):
            input_, output = i[0].to(device), i[1].to(device)
            attention_mask = torch.ones((input_.shape[0], 1, ws)).to(device)
            predict = model(input_, attention_mask)
            print(predict)
            print('-------------------------------------------------')
            print(output)

            loss = loss_fct(predict, output)

            print(loss.item())

            epoch_loss += input_.shape[0] * loss.item()
            count += input_.shape[0]

        print(f'average test loss at epoch {e}: {epoch_loss / count}')

        model.eval()
        eval_loss = 0
        count = 0
        predicts = []
        labels = []
        for i in tqdm(
                eval_loader,
                mininterval=200
        ):
            input_, output = i[0].to(device), i[1].to(device)
            attention_mask = torch.ones((input_.shape[0], 1, ws)).to(device)
            predict = model(input_, attention_mask)

            # print(predict.cpu().detach().numpy())
            # print(output.cpu().detach().numpy())
            # print(predict.shape)
            # quit()

            loss = loss_fct(predict, output)
            eval_loss += input_.shape[0] * loss.item()
            count += input_.shape[0]

        eval_loss = eval_loss / count

        print(f'total eval loss at epoch {e}: {eval_loss}')
        if eval_loss < best_eval_loss:
            best_eval_score = eval_loss
            torch.save({'model': model.state_dict()}, f'checkpoint/best_epoch{e}_loss_{round(best_eval_loss, 3)}.pt')
            print('saving better checkpoint')
