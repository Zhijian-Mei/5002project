import torch
from torch import nn
import torch.nn.functional as f

torch.set_default_dtype(torch.float32)
class MyModel(nn.Module):

    def __init__(self, args, input_size, device):
        super(MyModel, self).__init__()
        self.device = device
        self.args = args
        self.lstm_layers = 2
        self.bidirectional = True
        self.emb = nn.Linear(input_size, args.hidden_size).to(device)
        self.extract = BERT(args.hidden_size, args.hidden_size,device)
        # self.projectUp1 = nn.Linear(args.hidden_size,512).to(device)
        # self.projectUp2 = nn.Linear(512,1024).to(device)
        # self.out = nn.Linear(1024,1).to(device)
        # self.dropout = nn.Dropout(0.1)
        self.project = nn.LSTM(args.hidden_size, args.hidden_size, self.lstm_layers, batch_first=True,
                               bidirectional=self.bidirectional).to(device)
        self.out0 = nn.Linear(args.hidden_size * 2 if self.bidirectional else args.hidden_size, 1).to(device)
        # self.out1 = nn.ReLU()

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        input_tensor = self.emb(input_tensor)
        encoded = self.extract(input_tensor, attention_mask)
        # h0 = torch.zeros(self.lstm_layers,encoded.shape[0],  self.args.hidden_size).to(self.device)
        # c0 = torch.zeros(self.lstm_layers,encoded.shape[0], self.args.hidden_size).to(self.device)
        encoded, (hn, cn) = self.project(encoded)
        # encoded = self.projectUp1(encoded)
        # encoded = self.dropout(encoded)
        # encoded = self.projectUp2(encoded)
        output = self.out0(encoded).squeeze()
        # output = self.out1(output)
        return output



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


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=2, dropout=0):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):
    def __init__(self, dim_inp, dim_out, device,attention_heads=2, num_blocks = 2):
        super(BERT, self).__init__()
        self.module_list = [Encoder(dim_inp, dim_out, attention_heads).to(device) for _ in range(num_blocks)]

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        for module in self.module_list:
            encoded = module(input_tensor, attention_mask)

        return encoded
