import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 

# Generate synthetic data
t = np.linspace(0, 100, 1000)
y = np.sin(t) + 0.1*np.random.randn(len(t))

# Classes for the architecture
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len = 5000):
    super().__init__()
    pe = torch.zeros(max_len, d_model) # this creates the encoding matrix (max_len, d_model) 
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log10(10000.0)/d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe) # loads into to model and will not be updated on training

  def forward(self, x):
    x = x + self.pe[:, :x.size(1), :] # positional encoding update, tensor of form (batch, seq_len, d_model)
    return x
  
class TransformerSineDenoiser(nn.Module):
  def __init__(self, input_dim = 1, d_model = 64, nhead = 4, num_layers = 2, 
               dim_feedforward = 128):
    super().__init__()
    self.input_proj = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead = nhead, dim_feedforward=dim_feedforward)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.output_proj = nn.Linear(d_model,1)
  
  def forward(self, src):
    # src is the encoder vector, conventially used instead of x
    src = self.input_proj(src)
    src = self.pos_encoder(src)
    src = src.permute(1, 0, 2)
    out = self.transformer_encoder(src)
    out = out.permute(1, 0, 2) # undo the permutation above
    return self.output_proj(out)

