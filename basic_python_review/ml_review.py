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
  
  return torch.tensor(X), torch.tensor(y)

X_data, y_data = create_dataset(y, lookback=50)
print(f"Shapes: X: {X_data.shape}, y: {y_data.shape}")

class SineAnalyzer(nn.Module):
  pass 
