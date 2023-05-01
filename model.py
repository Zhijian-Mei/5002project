import torch
from torch import nn
import torch.nn.functional as f
from transformers import BertModel,BertConfig
class MyModel(nn.Module):

    def __init__(self,args,input_size):
        super(MyModel, self).__init__()
        self.emb = nn.Linear(input_size,args.hidden_size)
        self.extract = BERT(args.hidden_size,args.hidden_size)
        self.project = nn.Linear(args.hidden_size,1)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        input_tensor = self.emb(input_tensor)
        encoded = self.extract(input_tensor,attention_mask)
        encoded = self.project(encoded).squeeze()
        return encoded

    def predict(self,input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        input_tensor = self.emb(input_tensor)
        encoded = self.extract(input_tensor,attention_mask)
        encoded = self.project(encoded)



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

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor=None):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
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

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor=None):
        context = self.attention(input_tensor,attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4):
        super(BERT, self).__init__()
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor=None):
        encoded = self.encoder(input_tensor,attention_mask)

        return encoded