import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data as data 
import math 
import copy 

# Multihead attention
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    assert d_model % num_heads == 0        # this needs to happen as d_model should be divisible by the number of heads

    self.d_model   = d_model 
    self.num_heads = num_heads

    self.d_k = d_model // num_heads        # integer division to get the dimensionality of each key, this is for the scaled dot product

    # Establish weights for each of the channels (Query, Key, Value) and Output
    self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

  def scaled_dot_product_attention(self, Q, K, V, mask = None):
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # the (-2, -1) in transpose is made for switching the last 2 dimensions
    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask==0, -1e9)
    attention_probabilities = torch.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probabilities, V)
    return output 

  def split_heads(self, x):
    batch_size, seq_length, d_model = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

  def combine_heads(self, x):
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))

    attention_output = self.scaled_dot_product_attention(Q,K,V,mask)
    output = self.W_o(self.combine_heads(attention_output))
    return output




