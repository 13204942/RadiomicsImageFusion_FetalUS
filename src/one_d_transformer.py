import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_size, max_len=100):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, embed_size)
#         for pos in range(max_len):
#             for i in range(0, embed_size, 2):
#                 position = torch.tensor([[pos]], dtype=torch.float32)
#                 div_term = torch.pow(
#                     10000, (
#                         2 * (i // 2)) / torch.tensor(embed_size
#                     ).float()
#                 )
#                 self.encoding[pos, i] = torch.sin(
#                     position / div_term
#                 )
#                 self.encoding[pos, i + 1] = torch.cos(
#                     position / div_term
#                 )
#         self.encoding = self.encoding.unsqueeze(0)

#     def forward(self, x):
#         return x + self.encoding[:, : x.size(1), :].to(x.device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.size())
        print(self.pe.size())        
        x = x + self.pe[:x.size(0), :]
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class OneDTransformer(nn.Module):
    """
    Input:
        X: (batch, n_channel, n_length)
        Y: (batch)
        
    Output:
        out: (batch, n_classes)
        
    Pararmetes:
    """
    def __init__(self, n_classes, n_label, n_length, n_channel, vocab_size, num_embed, embed_size, nhead, dim_feedforward, dropout, activation, verbose=False ):
        super(OneDTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_label = n_label
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.embed_size = embed_size
        self.nhead = nhead
        self.n_length = n_length
        self.n_channel = n_channel
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.verbose = verbose

        self.embed1 = nn.Linear(self.n_channel, self.num_embed)
        # self.embed = nn.Linear(self.n_length, self.embed_size)
        # self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        fn_dim = int(self.embed_size/self.num_embed)
        self.embed2 = nn.Linear(self.num_embed, fn_dim)

        # self.pos_encoder = PositionalEncoding(self.embed_size, self.n_length)
        self.pos_encoder = PositionalEncoding(self.num_embed)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dense = nn.Linear(self.embed_size, self.vocab_size)
        # self.softmax = nn.Softmax(dim=2)

        self.mlp = nn.Sequential(
            # nn.Linear(self.vocab_size, 100),
            nn.Linear(self.n_length*self.vocab_size, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, self.n_label),
        )

    def forward(self, x):

        b, c, l = x.size()

        if self.verbose:
            print('input (batch, n_channel, n_length)', x.shape)
            # [16, 1, 95]

        # x = x.permute(2, 0, 1).squeeze()
        # x = x.squeeze()
        x = x.permute(0, 2, 1)
        if self.verbose:
            print('transpose (batch, n_length, n_channel)', x.shape)
            # [16, 95, 1]

        out = self.embed1(x)
        if self.verbose:
            print('embed1 (batch, n_length, n_features)', out.shape)
            # [16, 95, 64]
        
        out = out.unsqueeze(2)
        if self.verbose:
            print('embed1 (batch, n_length, 1, n_features)', out.shape)
            # [16, 95, 1, 64]

        out_p = self.embed2(out)
        # out_p = out_p.permute(0, 2, 1)
        if self.verbose:
            print('embed2 (batch, n_length, 1, n_features)', out_p.shape)
            # [16, 95, 1, 8]

        out_p = F.softmax(out_p, dim = 1)
        if self.verbose:
            print('softmax (batch, n_features, 1, n_length)', out_p.shape)
            # [16, 95, 1, 8]

        # out = out.permute(1, 0, 2)
        # out = self.pos_encoder(out)
        # if self.verbose:
        #     print('pos_encoder (n_length, batch, n_features)', out.shape)
        #     # [95, 16, 64]

        out = torch.matmul(out_p.permute(0, 1, 3, 2), out)
        if self.verbose:
            print('matmul (batch, n_length, n_features)', out.shape)
            # [16, 95, 64, 8]

        # print(torch.max(out))
        # print(torch.min(out))

        # out = out.permute(1, 0, 2)
        out = out.view(b, l, self.embed_size)
        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder (batch, n_length, n_features)', out.shape)
            # [16, 95, 512]

        # print(torch.max(out))
        # print(torch.min(out))

        # out = out.mean(1)
        # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax (batch, n_length, n_features)', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense (batch, n_length, vocab_size)', out.shape)

        # print(torch.max(out))
        # print(torch.min(out))

        # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax', out.shape)

        out = torch.flatten(out, start_dim=1)
        out = self.mlp(out)
        if self.verbose:
            print('mlp (batch, n_label)', out.shape)

        # print(torch.max(out))
        # print(torch.min(out))
        
        return out