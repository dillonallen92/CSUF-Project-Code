import torch
import torch.nn as nn 
from transformer_modules.feedforward import PositionwiseFeedForward

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, attention_impl):
    super().__init__()

    self.attention = attention_impl(d_model, nhead, dropout)
    self.attention_norm = nn.LayerNorm(d_model)
    self.attention_dropout = nn.Dropout(dropout)

    self.feedforward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
    self.feedforward_norm = nn.LayerNorm(d_model)
    self.feedforward_dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    # here we will use the attention implementation we wired above
    attention_output = self.attention(x, x, x)
    x = x + self.attention_dropout(attention_output)
    x = self.attention_norm(x)

    # pass through the feedforward net
    feedforward_output = self.feedforward(x)
    x = x + self.feedforward_dropout(feedforward_output)
    x = self.feedforward_norm(x)
    return x
