import pandas as pd 
import torch.nn as nn 
import torch.optim as optim 
from models import LSTM
from trainer import Trainer 
from data_utils import prep_data
from loss_functions import RMSELoss
from data_loader import combine_vf_fire_pop_data

def main():

  fire_data_path       = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv"
  vf_data_path         = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv"
  pop1_data_path       = "Project/data/cali_county_pop_2000_2010.csv" 
  pop2_data_path       = "Project/data/cali_county_pop_2010_2020.csv"
  start_year           = "2006"
  end_year             = "2015"
  county_name          = "Tulare"
  source_column_labels = ['Fire Incident Count', 'VF Case Count', 'Population']
  split_frac           = .85
  lookback             = 6
  input_size           = 3
  hidden_size          = 64
  num_layers           = 2
  dropout              = 0.2
  learning_rate        = 0.001
  epochs               = 300
  weight_decay         = 1e-5
  title_text           = "(Population Added)"

  wf_vf_pop_df = combine_vf_fire_pop_data(pop1_data_path, pop2_data_path, vf_data_path, fire_data_path, county_name,
                                          start_year, end_year, bInterp=False)

  X_train, y_train, X_test, y_test, scaler = prep_data(wf_vf_pop_df, source_column_labels, 
                                                        split_frac, lookback, b_scaler=True)

  # Able to be swapped out with other models (I hope)
  model     = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
  criterion = RMSELoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)  

  trainer = Trainer(model = model, criterion = criterion, optimizer = optimizer, scaler = scaler)
  trainer.train(X_train, y_train, X_test, y_test, epochs)

  y_pred, y_true = trainer.evaluate(X_test, y_test)
  trainer.visualize_results(y_true, y_pred, county_name, title_text = title_text, show_plot=True, save_fig = True)

if __name__ == "__main__":
  main()