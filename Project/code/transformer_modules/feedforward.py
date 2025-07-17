import torch
import torch.nn as nn 

class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, dim_feedforward, dropout):
    super().__init__()
    self.fc1 = nn.Linear(d_model, dim_feedforward)
    self.fc2  = nn.Linear(dim_feedforward, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # Here we will go through FC1 to ReLU, hit a dropout, go to FC2, 
    # then another dropout
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x 