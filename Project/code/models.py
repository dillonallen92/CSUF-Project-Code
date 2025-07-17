import torch
import torch.nn as nn
from transformer_modules.encoder import TransformerEncoder
from transformer_modules.positional import PositionalEncoding
from transformer_modules.attention import MultiHeadAttention

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, dropout, num_layers = 1):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout=dropout)
    self.linear = nn.Linear(hidden_size, 1) # single output (predicting VF case rate)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    last_out = out[:,-1,:]
    return self.linear(last_out)

class TransformerModel(nn.Module):
  def __init__(self, input_size, d_model = 32, nhead = 2, num_layers = 1, dim_feedforward = 64, 
               dropout = 0.2, attention_impl = MultiHeadAttention):

    super().__init__()
    self.input_proj = nn.Linear(input_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, dropout)

    self.encoder = nn.ModuleList([
      TransformerEncoder(d_model, nhead, dim_feedforward, dropout, attention_impl)
      for _ in range(num_layers)
      ])
    self.norm     = nn.LayerNorm(d_model)

    self.head     = nn.Linear(d_model, 1) # output value
  
  def forward(self, x):
    x = self.input_proj(x)
    x = self.pos_encoder(x)

    for layer in self.encoder:
      x = layer(x)
    
    x = self.norm(x)
    x = x[:, -1, :]
    return self.head(x)