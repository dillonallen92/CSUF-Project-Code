import pandas as pd 
import torch.nn as nn 
import torch.optim as optim 
from models import LSTM
from trainer import Trainer 
from data_utils import prep_data
from loss_functions import RMSELoss
from data_loader import combine_vf_wildfire_data

def main():
  # data import and setup parameters
  # Change these values here, everything else should work without user input
  # --- Load Data ---
  fire_data_path = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv"
  vf_data_path = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv"
  county_name = "Fresno"
  vf_fire_data = combine_vf_wildfire_data(fire_path=fire_data_path, vf_cases_path=vf_data_path, county_name=county_name)
  source_column_labels = ['Fire Incident Count', 'VF Case Count']
  split_frac           = .85
  lookback             = 8
  input_size           = 2
  hidden_size          = 64
  num_layers           = 2
  dropout              = 0.2
  learning_rate        = 0.001
  epochs               = 300

  X_train, y_train, X_test, y_test, scaler = prep_data(vf_fire_data, source_column_labels, 
                                                       split_frac, lookback, b_scaler=True)
  
  # Able to be swapped out with other models (I hope)
  model     = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
  criterion = RMSELoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)  

  trainer = Trainer(model = model, criterion = criterion, optimizer = optimizer, scaler = scaler)
  
  trainer.train(X_train, y_train, X_test, y_test, epochs)
  y_pred, y_true = trainer.evaluate(X_test, y_test)
  trainer.visualize_results(y_true, y_pred, county_name, show_plot=True, save_fig = True)

if __name__ == "__main__":
  main()