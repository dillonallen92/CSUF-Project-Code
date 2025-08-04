import torch
import torch.nn as nn 
import math 

# This script will hold a multitude of attention classes like 
# the default multihead attention class and probsparse

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by number of heads"

    self.d_model = d_model
    self.num_heads  = num_heads 
    self.d_k     = d_model // num_heads 

    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)

    self.out_proj = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)
  
  def scaled_dot_product(self, Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    return torch.matmul(attention_weights, V)

  def split_heads(self, x):
    batch_size, seq_len, _ = x.size()
    x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
    return x.transpose(1,2)

  def combine_heads(self, x):
    batch_size, _, seq_len, _ = x.size()
    x = x.transpose(1,2).contiguous()
    return x.view(batch_size, seq_len, self.d_model)
  
  def forward(self, query, key, value):
    Q = self.split_heads(self.W_q(query))
    K = self.split_heads(self.W_k(key))
    V = self.split_heads(self.W_v(value))

    attention_output = self.scaled_dot_product(Q, K, V)
    concatenate_output = self.combine_heads(attention_output)
    return self.out_proj(concatenate_output)