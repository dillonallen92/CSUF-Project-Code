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

# First pass of LSTM Class
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers = 1):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
    self.linear = nn.Linear(hidden_size, 1) # single output (predicting VF case rate)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    last_out = out[:,-1,:]
    return self.linear(last_out)


# TODO: Results Plotting Function
def visualize_results(data):
  pass

if __name__ == "__main__":  
  print("Valley Fever Prediction LSTM")
  ########################
  #     Input Values    #
  #######################
  source_column_labels = ['Fire Incident Count', 'VF Case Count']
  split_frac           = .85
  lookback             = 9
  input_size           = 2
  hidden_size          = 64
  num_layers           = 1
  learning_rate        = 0.001
  epochs               = 300

  ########################
  #   Dataset Creation   #
  ########################
  print("Creating Training and Testing Data...")
  X_train, y_train, X_test, y_test, scaler = prep_data(vf_fire_data, source_column_labels, 
                                               split_frac = split_frac, lookback = lookback, 
                                               b_scaler = True)


  #Create the LSTM, feed data into it, plot results
  # On first pass, do everything inside main... will refactor into functions/classes

  #######################
  #   Model Creation   #
  ######################
  print("Create the model...")
  model = LSTM(input_size=input_size, hidden_size = hidden_size, num_layers = num_layers)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  ######################
  #   Model Training   #
  ######################
  print("Train the model...")
  
  for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss   = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
      model.eval()
      val_loss = criterion(model(X_test), y_test)
      print(f"Epoch: {epoch + 1}/{epochs}- Train Loss: {loss.item():.4f}, Test Loss: {val_loss.item():.4f}")
  
  ##############################
  #   Generating Predictions   #
  ##############################
  print("generating predictions...")
  model.eval()
  pred_scaled = model(X_test).detach().numpy()

  # Reconstruct full array to inverse transform
  dummy = np.zeros((len(pred_scaled), 2))
  dummy[:, 1] = pred_scaled[:, 0]  # VF case prediction in 2nd column
  vf_pred = scaler.inverse_transform(dummy)[:, 1]  # inverse only VF

  # Compare to true values
  true_scaled = y_test.numpy()
  dummy[:, 1] = true_scaled[:, 0]
  vf_true = scaler.inverse_transform(dummy)[:, 1]

  print("Predicted (scaled):", pred_scaled[:5])
  print("Predicted (inverted):", vf_pred[:5])
  print("True (inverted):", vf_true[:5])

  ########################
  #   Plotting Results   #
  ########################
  print("plotting results...")
  # Create a figure
  plt.figure(figsize=(12, 6))

  # Plot true VF case counts
  plt.plot(vf_true, label="True VF Cases", linewidth=2)

  # Plot predicted VF case counts
  plt.plot(vf_pred, label="Predicted VF Cases", linestyle="--", linewidth=2)

  # Add labels and legend
  plt.title("Predicted vs True Valley Fever Case Counts (Test Set)", fontsize=14)
  plt.xlabel("Time (Months)", fontsize=12)
  plt.ylabel("VF Case Count", fontsize=12)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
