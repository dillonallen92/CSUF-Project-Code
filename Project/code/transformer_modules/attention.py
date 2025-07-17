import torch
import torch.nn as nn 
import math 

# This script will hold a multitude of attention classes like 
# the default multihead attention class and probsparse

class MultiHeadAttentionCode(nn.Module):
  def __init__(self, d_model, nheads, dropout):
    super().__init__()
    assert d_model % nheads == 0, "d_model must be divisible by number of heads"

    self.d_model = d_model
    self.nheads  = nheads 
    self.d_k     = d_model // nheads 

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