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
from evaluation import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-ws', type=int, default=288)
    parser.add_argument('-debug', type=int, default=0)
    parser.add_argument('-uid', type=str, default='dummy')
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
    torch.set_default_dtype(torch.float32)
    subset = ['TurbID', 'Wspd', 'Wdir', 'Patv']

    uid = args.uid

    root_name = f'experiment_{uid}'

    final_score = 0

    in_file_root = f'data/final_phase_test/infile'
    out_file_root = f'data/final_phase_test/outfile'
    for input_index in range(1,143):
        current_index = f'{input_index:04d}'
        in_file = f'{in_file_root}/{current_index}.csv'
        current = pd.read_csv(in_file)
        dfs = list(current[subset].groupby('TurbID'))
        for item in dfs:
            id = item[0]
            df = item[1]

            # prepare checkpoint folder
            folder_name = f'checkpoint/{root_name}/turbine_{id}'

            checkpoint = torch.load(f'{folder_name}/best_model.pt')
            model = MyModel(args, len(subset) - 1, device)
            model.load_state_dict(checkpoint['model'])

            train = df

            train_size = int(len(train) * 0.8)
            train_set, eval_set = train[:train_size],train[train_size:]

            print(f'loading data for turbine {id}')
            # train_dataset = MyDataset(train_set, ws=ws)
            eval_dataset = MyDataset(eval_set, ws=ws)

            # print('number of train sample: ',len(train_dataset))
            print('number of eval sample: ',len(eval_dataset))

            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

            loss_fct = nn.MSELoss()
            count = 0
            eval_loss = 0

            print('start testing')
            model.eval()
            with torch.no_grad():
                for i in tqdm(
                        eval_loader,
                        # mininterval=200
                ):
                    input_, output = i[0].to(device), i[1].to(device)
                    attention_mask = torch.ones((input_.shape[0], 1, ws)).to(device)
                    predict = model(input_, attention_mask)

                    score = score_t_abnormal(predict.cpu().numpy(),output.cpu().numpy()[0])
                    print(score)
                    loss = loss_fct(predict, output)
                    eval_loss += input_.shape[0] * loss.item()
                    count += input_.shape[0]

                eval_loss = eval_loss / count

