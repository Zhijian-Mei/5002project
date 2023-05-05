import argparse
import os
import time

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
from evaluation import score_t


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-ws', type=int, default=288)
    parser.add_argument('-debug', type=int, default=0)
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-epoch', type=int, default=20)
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
        df = pd.read_csv('data/clean_fill_data.csv')[:100000]
    else:
        df = pd.read_csv('data/clean_fill_data.csv')

    subset = ['TurbID', 'Wspd', 'Wdir', 'Patv']
    df = df[subset]

    dfs = list(df.groupby('TurbID'))

    import uuid
    uid = uuid.uuid4()

    root_name = f'experiment_{uid}'
    os.system(f'mkdir checkpoint/{root_name}')
    for item in dfs:
        id = item[0]
        df = item[1]

        # prepare checkpoint folder
        folder_name = f'checkpoint/{root_name}/turbine_{id}'
        os.system(f'mkdir {folder_name}')

        model = MyModel(args, len(subset) - 1, device)

        train = df

        train_size = int(len(train) * 0.8)
        train_set, eval_set = train[:train_size],train[train_size:]

        print(f'loading data for turbine {id}')
        train_dataset = MyDataset(train_set, ws=ws)
        eval_dataset = MyDataset(eval_set, ws=ws)

        print('number of train sample: ',len(train_dataset))
        print('number of eval sample: ',len(eval_dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        epoch = args.epoch
        global_step = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fct = nn.MSELoss()
        best_eval_loss = np.inf
        count = 0
        best_model = None
        print('start training')
        for e in range(epoch):
            epoch_loss = 0
            model.train()
            for i in tqdm(
                    train_loader,
                    mininterval=200
            ):
                input_, output = i[0].to(device), i[1].to(device)
                attention_mask = torch.ones((input_.shape[0], 1, ws)).to(device)
                predict = model(input_, attention_mask)
                loss = loss_fct(predict, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                epoch_loss += input_.shape[0] * loss.item()
                count += input_.shape[0]

            print(f'average train loss at epoch {e}: {epoch_loss / count}')

            model.eval()
            eval_loss = 0
            count = 0
            predicts = []
            labels = []
            with torch.no_grad():
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

                print(f'average eval loss at epoch {e}: {eval_loss}')
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model = model
                    torch.save({'model': model.state_dict()}, f'{folder_name}/best_epoch{e}_loss_{round(best_eval_loss, 3)}.pt')
                    print('saving better checkpoint')
        torch.save({'model': best_model.state_dict()},
                   f'{folder_name}/best_model.pt')
        print(f'finish turbine {id}')
        print('train next turbine')