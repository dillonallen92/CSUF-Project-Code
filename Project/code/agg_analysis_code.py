import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import torch 
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from models import LSTM, TransformerModel 
from trainer_new import TrainerNewNew
from loss_functions import RMSELoss

def prep_data(csv_path:str) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  df["Year-Month"] = pd.to_datetime(df["Year-Month"])
  df = df.set_index('Year-Month')
  return df

def feature_target_vectors(df: pd.DataFrame, targetCol:str  = "VFRate") -> tuple[pd.DataFrame, pd.Series]:
  features = df.drop(columns = [targetCol])
  target  = df[targetCol]
  return features, target

def create_sequences(X:pd.DataFrame, y:pd.Series, seq_length:int = 12) -> tuple[np.array, np.array]:
  Xs, ys = [], []
  for idx in range(len(X) - seq_length):
    Xs.append(X[idx:idx+seq_length])
    ys.append(y[idx+seq_length])
  return np.array(Xs), np.array(ys)

def transform_np_to_tensor(x:np.array) -> torch.Tensor:
  return torch.tensor(x, dtype=torch.float32)

def split_data(X: np.array, y: np.array, split_frac:float = .80) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  split_idx = int(len(X)*split_frac)
  X = transform_np_to_tensor(X)
  y = transform_np_to_tensor(y)
  X_train, X_test = X[:split_idx], X[split_idx:]
  y_train, y_test = y[:split_idx], y[split_idx:]
  return X_train, X_test, y_train, y_test 

def transform_Data(X_train: torch.Tensor, X_test: torch.Tensor, 
                  y_train: torch.Tensor, y_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor, MinMaxScaler]:
  # Reshape the feature vectors for scaling
  X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
  X_test_reshaped  = X_test.reshape(-1, X_test.shape[-1])

  # create the scaler classes
  scaler_X = MinMaxScaler()
  scaler_Y = MinMaxScaler()

  # scale and transfomer the feature and target vectors
  X_train_scaled = scaler_X.fit_transform(X_train_reshaped.numpy())
  X_test_scaled  = scaler_X.transform(X_test_reshaped.numpy())
  y_train_scaled = scaler_Y.fit_transform(y_train.reshape(-1, 1).numpy())
  y_test_scaled  = scaler_Y.transform(y_test.reshape(-1, 1).numpy())

  # convert back to tensors and return
  X_train_final = transform_np_to_tensor(X_train_scaled.reshape(X_train.shape))
  X_test_final  = transform_np_to_tensor(X_test_scaled.reshape(X_test.shape))
  y_train_final = transform_np_to_tensor(y_train_scaled)
  y_test_final  = transform_np_to_tensor(y_test_scaled)

  return X_train_final, X_test_final, y_train_final, y_test_final, scaler_Y



if __name__ == "__main__":
  print("--- Aggregate Data Analysis ---")

  # Parameters to change
  data_path     = "Project/data/Fresno_Aggregate.csv"
  seq_length    = 12
  split_frac    = 0.8
  lookback      = 12
  hidden_size   = 32
  num_layers    = 2
  dropout       = 0.2
  learning_rate = 0.001
  epochs        = 300
  weight_decay  = 1e-5

  # Process
  df   = prep_data(data_path)
  features, target = feature_target_vectors(df, targetCol="VFRate")
  features, target = create_sequences(features, target, seq_length=seq_length)
  X_train, X_test, y_train, y_test = split_data(features, target, split_frac = split_frac)
  
