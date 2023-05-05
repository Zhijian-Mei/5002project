import argparse

import numpy as np
import pandas as pd
import torch
from torch import cuda, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

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
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float64)
    subset = ['TurbID', 'Wspd', 'Wdir', 'Patv']
    uid = args.uid

    root_name = f'experiment_{uid}'

    score_per_df = []
    mean_df = pd.read_csv('data/mean_record.csv')
    std_df = pd.read_csv('data/std_record.csv')
    in_file_root = f'data/final_phase_test/infile'
    out_file_root = f'data/final_phase_test/outfile'
    print('start testing')
    for k in trange(1,143):
        score = 0
        current_index = f'{k:04d}'
        in_file = f'{in_file_root}/{current_index}in.csv'
        out_file = f'{out_file_root}/{current_index}out.csv'
        current_in = pd.read_csv(in_file)
        current_out = pd.read_csv(out_file)
        in_dfs = list(current_in[subset].groupby('TurbID'))
        out_dfs = list(current_out[subset].groupby('TurbID'))
        for df_index in range(len(in_dfs)):
            item_in = in_dfs[df_index]
            item_out = out_dfs[df_index]
            id = item_in[0]
            df_in = item_in[1]
            df_out = item_out[1]
            current_df_in = df_in[len(df_in)-288:]
            current_df_out = df_out
            input_ = current_df_in.drop(columns=['Patv','TurbID'])
            output_ = current_df_out['Patv']
            input_ = input_.fillna(input_.mean()).values
            output_ = output_.fillna(output_.mean()).values

            current_mean = mean_df[mean_df.TurbID == id][['Wspd','Wdir']].values[0]
            current_std = std_df[std_df.TurbID == id][['Wspd','Wdir']].values[0]
            current_mean = [current_mean for _ in range(288)]
            current_std = [current_std for _ in range(288)]
            normalize_input = (input_ - current_mean)/current_std

            patv = current_df_in[['Patv']].fillna(current_df_in[['Patv']].mean()).values
            patv = np.where(patv < 0 , 0 , patv)
            normalize_input = np.concatenate((normalize_input,patv),axis=1)
            # prepare checkpoint folder
            folder_name = f'checkpoint/{root_name}/turbine_{id}'

            checkpoint = torch.load(f'{folder_name}/best_model.pt',map_location=device)
            model = MyModel(args, len(subset) - 1, device)
            model.load_state_dict(checkpoint['model'])

            model.eval()
            with torch.no_grad():
                input_ = torch.from_numpy(normalize_input).unsqueeze(0)
                output_ = torch.from_numpy(output_)
                attention_mask = torch.ones((1, 1, ws)).to(device)
                predict = model(input_, attention_mask)
                score_t = score_t_abnormal(predict.numpy(),output_.numpy())
                print(score_t)
                score+=score_t
        score_per_df.append(score)



