import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import torch_geometric.nn as pygnn
import math
from masking import FullMask, LengthMask


class ScaledDotProAttention(nn.Module):
    def __init__(self, dropout=0):
        super(ScaledDotProAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, mask=None):
        # input shape [batch_szie, seq_length, embedding_dim]
        attention = torch.bmm(q, k.transpose(1,2))

        # attention shape is [batch_size, seq_lenth, seq_length]
        if scale is not None:
            attention = attention * scale
        if mask is not None:
            attention = attention.masked_fill_(mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

def padding_mask(seq):
    # seq shape is [batch_szie, seq_length]
    length = seq.shape[1]
    mask = seq.eq(0)
    mask = mask.unsqueeze(1).expand(-1, length, -1)
    return mask

class FFN(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        # input shape is [batch_size, seq_len, embedding_dim]
        output = self.fc1(input)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        output = self.fc2(output)

        return output

class TransformerEncoderLayer(nn.Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        # print("attn_mask.dtype")
        # print(attn_mask.dtype)
        # attn_mask = attn_mask or FullMask(L, device=x.device)

        # length_mask = length_mask or \
        #     LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=attn_mask
        )
        
        x = x + self.dropout(attn_output)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)
    

class Encoder_Layer(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, num_heads=8, dropout=0):
        super(Encoder_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.calculate_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_q.weight, gain=1)
        self.calculate_k = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_k.weight, gain=1)
        self.calculate_v = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_v.weight, gain=1)
        self.MultiheadAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal(self.fc1.weight, gain=1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal(self.fc2.weight, gain=1)
        
        self.dropout2 = nn.Dropout(p=dropout)

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.LeakyReLU()

    def forward(self, querys, keys, values, mask=None):
        # print("query.shape")        #[1, 29601, 32]
        # print(querys.shape)
        # print("keys.shape")         #[1, 29601, 32]
        # print(keys.shape)
        # print("values.shape")       #[1, 29601, 32]
        # print(values.shape)
        query = self.calculate_q(querys).transpose(0, 1).contiguous()
        key = self.calculate_k(keys).transpose(0, 1).contiguous()
        value = self.calculate_v(values).transpose(0, 1).contiguous()

        output, _ = self.MultiheadAttention(query, key, value, key_padding_mask=mask)
        output = output.transpose(0, 1).contiguous()

        output = querys + self.dropout(output)
        output = self.layer_norm(output)

        output = self.fc(output)
        tmp = output

        output = self.fc2(self.dropout2(self.relu2(self.fc1(output))))
        output = tmp + self.dropout2(output)
        output = self.layer_norm2(output)

        # print("output.shape")
        # print(output.shape)
        
        return output



class GNNTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head=8, num_layers=2, dropout=0, device='cpu'):
        super(GNNTransformer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.transformer_encoders = nn.ModuleList()
        self.gnn_encoders = nn.ModuleList()

        self.transformer_encoders.append(Encoder_Layer(self.in_dim, self.hidden_dim, self.num_head, self.dropout))
        for i in range(self.num_layers):
            self.gnn_encoders.append(pygnn.SAGEConv(self.hidden_dim, self.hidden_dim))
            # self.gnn_encoders.append(pygnn.GATConv(self.hidden_dim, self.hidden_dim, heads=1))
            self.transformer_encoders.append(Encoder_Layer(self.hidden_dim, self.hidden_dim, self.num_head, self.dropout))

        self.fc = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.xavier_normal(self.fc.weight, gain=1)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # stdv = 1.0 / math.sqrt(self.embedding_dim)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, stdv)
        
        
    def forward(self, graph, x, sample_index_m, weights=None, mask=None):
        edge_index = graph.edge_index
        
        samples = sample_index_m

        sample_features = F.embedding(samples, x)
        center_features = torch.unsqueeze(x, dim=1)
        
        for idx, transformer_encoder in enumerate(self.transformer_encoders):
            output = transformer_encoder(center_features, sample_features, sample_features, mask)
            # print(center_features.size(), sample_features.size(), output.size())
            if idx < self.num_layers:
                gnn_encoder = self.gnn_encoders[idx]
                output = torch.squeeze(output)
                output = gnn_encoder(output, edge_index)
                
                sample_features = F.embedding(samples, output)
                center_features = torch.unsqueeze(output, dim=1)

        output = self.fc(output)
        output = torch.squeeze(output)

        return output
