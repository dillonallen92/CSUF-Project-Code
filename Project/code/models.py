import torch.nn as nn

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers = 1):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
    self.linear = nn.Linear(hidden_size, 1) # single output (predicting VF case rate)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    last_out = out[:,-1,:]
    return self.linear(last_out)