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


# Creating a Trainer Class to contain Training/Testing/Visualizing
class Trainer:
  """
  Trainer Class: A class that contains the training, testing, and visualization functions.
  """
  def __init__(self, model, criterion, optimizer, scaler):
    """
    Initialize the class. Takes in a model, crtierion for loss, optimizer, scaler.
    
    Inputs:
      - Model: Neural Network model
      - Criterion: Loss function (Typically MSELoss for time series, may look into more)
      - Optimizer: Optimizer with learning rate added. Typically using Adam
      - Scaler: MinMaxScaler scaler value, used for inverse transform to get actual data back
    """
    self.model     = model 
    self.criterion = criterion 
    self.optimizer = optimizer
    self.scaler    = scaler
  
  def train(self, X_train, y_train, X_test, y_test, epochs):
    """
    Training Loop Function

    Inputs:
      - X_train: Training matrix X
      - y_train: training target vector y
      - X_test: Test matrix X
      - y_test: Target vector y
      - epochs: Number of Epochs to train
    """
    for epoch in range(epochs):
      self.model.train()
      output = self.model(X_train)
      loss   = self.criterion(output, y_train)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if epoch % 10 == 0:
        self.model.eval()
        val_loss = self.criterion(self.model(X_test), y_test)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss {loss.item():.4f}, Testing Loss {val_loss.item():.4f}")
    
  def evaluate(self, X_test, y_test):
      """
      Evaluation Loop. Evaluates the model and generates predictions.

      Inputs:
        - X_test: Test matrix X
        - y_test: Test target vector y
      
      Outputs:
        - y_pred: predicted target vector from the model using X_test
        - y_true: True target vector (y_test)
      """
      self.model.eval()
      with torch.no_grad():
        preds = self.model(X_test).detach().numpy()

        dummy = np.zeros((len(preds), 2))
        dummy[:, 1] = preds[:, 0]
        vf_pred = self.scaler.inverse_transform(dummy)[:,1]

        dummy[:,1] = y_test.numpy().flatten()
        vf_true = self.scaler.inverse_transform(dummy)[:,1]
      
      return vf_pred, vf_true
  
  def visualize_results(self, true, pred):
    """
    Function to visualize the prediction vs true (test) vector

    Inputs:
      - True: True data (test or validation target vector)
      - Pred: Prediction data from the model evaluation function
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True Values")
    plt.plot(pred, label = "Predicted Values", linestyle="--")
    plt.title("LSTM True vs Predicted Valley Fever Case Rates")
    plt.xlabel("Months")
    plt.ylabel("Case Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
  # data import and setup parameters
  vf_fire_data = pd.read_csv("Project/data/fire_vs_vf.csv")
  source_column_labels = ['Fire Incident Count', 'VF Case Count']
  split_frac           = .85
  lookback             = 9
  input_size           = 2
  hidden_size          = 64
  num_layers           = 1
  learning_rate        = 0.001
  epochs               = 300

  X_train, y_train, X_test, y_test, scaler = prep_data(vf_fire_data, source_column_labels, 
                                                       split_frac, lookback, b_scaler=True)
  
  # Able to be swapped out with other models (I hope)
  model     = LSTM(input_size, hidden_size, num_layers)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)  

  trainer = Trainer(model = model, criterion = criterion, optimizer = optimizer, scaler = scaler)
  
  trainer.train(X_train, y_train, X_test, y_test, epochs)
  y_pred, y_true = trainer.evaluate(X_test, y_test)
  trainer.visualize_results(y_true, y_pred)


if __name__ == "__main__":
  main()
