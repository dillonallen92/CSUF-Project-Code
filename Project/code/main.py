import pandas as pd 
import torch.nn as nn 
import torch.optim as optim 
from models import LSTM
from trainer import Trainer 
from data_utils import prep_data

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