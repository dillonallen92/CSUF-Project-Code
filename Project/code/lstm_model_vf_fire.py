# This code is meant to draft a LSTM model for the 
# valley fever 

# Package Imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from sklearn.preprocessing import MinMaxScaler

# data import
vf_fire_data = pd.read_csv("Project/data/fire_vs_vf.csv")

# Helper Functions

def transform_tensor(X,y):
  '''
  Converts data into Torch tensors
  
  Inputs:
    - X: input data
    - y: Target Data
  
  Outputs:
    - X_tensor: X transformed to torch tensor as float32
    - y_tensor: y transformed to torch tensor as float32
  '''
  X_tensor = torch.tensor(X, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

  return X_tensor, y_tensor

def train_test_split(data, split_frac):
  '''
  Function that computes the train/test split

  Inputs:
    - data: Dataframe to be processed
    - split_frac: fraction for training (test = 1 - split_frac)
  
  Outputs:
    - data_train: Training data up to split_idx
    - data_test: test data from split_idx til end
  '''
  split_idx  = int(len(data)*split_frac)
  data_train = data[:split_idx]
  data_test  = data[split_idx:]

  return data_train, data_test 

def prep_data(data, data_col_labels, split_frac, lookback, b_scaler=True):
  '''
  This function prepares the data to be passed into a LSTM
  or Transformer model. 

  Inputs
    - data: pandas dataframe, this will need to be redone based off size
    - split_frac: Train/Test split fraction
    - lookback: Lookback window for the time series
    - b_scalar: Boolean flag to use MinMaxScalar or not (probably always true)
  
  Outputs
    - X_train, y_train, X_test, y_test tensors prepared for analysis
  '''

  # First create numpy arrays then transform into torch tensors
  data = data.sort_values("Month").reset_index(drop=True)
  if b_scaler:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[data_col_labels])
    data   = scaled
  
  # Prep data with lookback window
  X_all, y_all = [], []
  N = len(data_col_labels)
  for i in range(len(data) - lookback):
    X_all.append(data[i:i+lookback, 0:N])
    y_all.append(data[i+lookback, 1])
  
  # Convert to numpy arrays
  X_all = np.array(X_all)
  y_all = np.array(y_all)

  # Create the train/test data
  X_train, X_test = train_test_split(X_all, split_frac)
  y_train, y_test = train_test_split(y_all, split_frac)

  X_train, y_train = transform_tensor(X_train, y_train)
  X_test, y_test   = transform_tensor(X_test, y_test)

  # Return the data
  if b_scaler:
    return X_train, y_train, X_test, y_test, scaler
  else:
    return X_train, y_train, X_test, y_test

# TODO: LSTM Class
class LSTM(nn.Module):
  pass


# TODO: Results Plotting Function
def visualize_results(data):
  pass

if __name__ == "__main__":  
  source_column_labels = ['Fire Incident Count', 'VF Case Count']
  split_frac = .70
  lookback   = 4
  X_train, y_train, X_test, y_test = prep_data(vf_fire_data, source_column_labels, 
                                               split_frac = split_frac, lookback = lookback, 
                                               b_scaler = True)


  # TODO: Create the LSTM, feed data into it, plot results