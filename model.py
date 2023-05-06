import random

import torch
from torch import nn
import torch.nn.functional as f

torch.set_default_dtype(torch.float32)
# class MyModel(nn.Module):
#
#     def __init__(self, args, input_size, device):
#         super(MyModel, self).__init__()
#         self.device = device
#         self.args = args
#         self.lstm_layers = 2
#         self.bidirectional = True
#         self.emb = nn.Linear(input_size, args.hidden_size).to(device)
#         self.extract = BERT(args.hidden_size, args.hidden_size,device)
#         self.projectUp1 = nn.Linear(args.hidden_size,2).to(device)
#         self.relu0 = nn.ReLU()
#         self.project = nn.Linear(2,1).to(device)
#         # self.project = nn.LSTM(args.hidden_size, args.hidden_size, self.lstm_layers, batch_first=True,
#         #                        bidirectional=self.bidirectional).to(device)
#         # self.out = nn.Linear(args.hidden_size * 2 if self.bidirectional else args.hidden_size, 1).to(device)
#         self.out = nn.ReLU()
#
#     def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
#         encoded = self.emb(input_tensor)
#         # encoded = self.extract(input_tensor, attention_mask)
#         # h0 = torch.zeros(self.lstm_layers,encoded.shape[0],  self.args.hidden_size).to(self.device)
#         # c0 = torch.zeros(self.lstm_layers,encoded.shape[0], self.args.hidden_size).to(self.device)
#         # encoded, (hn, cn) = self.project(encoded)
#         encoded = self.projectUp1(encoded)
#         encoded = self.relu0(encoded)
#         encoded = self.project(encoded)
#         output = self.out(encoded).squeeze()
#         # output = self.out1(output)
#         return output


class MyModel(nn.Module):
    def __init__(self,args,input_size, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(input_size=input_size,embedding_size=args.hidden_size,hidden_size=args.hidden_size,n_layers=2).to(device)
        self.decoder = Decoder(output_size=1,embedding_size=args.hidden_size,hidden_size=args.hidden_size,n_layers=2).to(device)


    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our argoverse motion forecasting dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]


        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)
        print(hidden.shape)

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :]
        print(x.shape)
        print(decoder_input.shape)
        quit()
        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs

class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 32,
                 hidden_size = 32,
                 n_layers = 2,
                 dropout = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = f.relu(self.linear(x))

        output, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 output_size=1,
                 embedding_size=32,
                 hidden_size=32,
                 n_layers=2,
                 dropout=0.1):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0)

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(f.relu(self.embedding(x)))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hidden size]
        # hidden = [n layers, batch size, hidden size]
        # cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell
class AttentionHead(nn.Module):

    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        attention_mask = attention_mask.bool()
        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


# class Encoder(nn.Module):
#
#     def __init__(self, dim_inp, dim_out, attention_heads=2, dropout=0):
#         super(Encoder, self).__init__()
#
#         self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)  # batch_size x sentence size x dim_inp
#         self.feed_forward = nn.Sequential(
#             nn.Linear(dim_inp, dim_out),
#             nn.Dropout(dropout),
#             nn.GELU(),
#             nn.Linear(dim_out, dim_inp),
#             nn.Dropout(dropout)
#         )
#         self.norm = nn.LayerNorm(dim_inp)
#
#     def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
#         context = self.attention(input_tensor, attention_mask)
#         res = self.feed_forward(context)
#         return self.norm(res)
#

class BERT(nn.Module):
    def __init__(self, dim_inp, dim_out, device,attention_heads=1, num_blocks = 1):
        super(BERT, self).__init__()
        self.module_list = [Encoder(dim_inp, dim_out, attention_heads).to(device) for _ in range(num_blocks)]

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        for module in self.module_list:
            encoded = module(input_tensor, attention_mask)

        return encoded
