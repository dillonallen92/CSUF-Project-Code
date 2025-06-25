import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch
import torch.nn as nn 
import torch.optim as optim 

# Flags for certain things to run
DIAGNOSTIC_PLOT = 0

# create the artificial data
t = np.linspace(0, 100, 10000)
y = np.sin(t) + 0.1*np.random.randn(len(t))

# plot the artificial data for now
if DIAGNOSTIC_PLOT:
  plt.plot(t, y)
  plt.show()

def create_dataset(series, lookback = 50):
  '''
  This function is responsible for creating the tensors that will be used in the LSTM
  from the dataset.
  '''

  X, y = [], [] 
  for i in range(len(series) - lookback):
    X.append(series[i:i+lookback])
    y.append(series[i+1:i+lookback+1])
  
  return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_frac = .70
train_idx = int(len(y)*train_frac)
X_train, y_train = create_dataset(y[:train_idx], lookback=50)
X_test, y_test   = create_dataset(y[train_idx:], lookback = 50)
print(f"Train Shape: X: {X_train.shape}, y: {y_train.shape}\nTest Shape: X: {X_test.shape}, y: {y_test.shape}")

# Create SineAnalyzer class that has the LSTM architecture
class SineAnalyzer(nn.Module):
  
  def __init__(self, input_size, hidden_size, num_layers):
    super().__init__()
    self.lstm = nn.LSTM(input_size= input_size,
                        hidden_size= hidden_size,
                        num_layers= num_layers,
                        batch_first= True)
    self.linear = nn.Linear(hidden_size, input_size)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    out    = self.linear(out)
    return out 


