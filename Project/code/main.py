import pandas as pd 
import torch.nn as nn 
import torch.optim as optim 
from models import LSTM, TransformerModel
from trainer import Trainer 
from data_utils import prep_data
from loss_functions import RMSELoss
from data_loader import combine_vf_fire_pop_data, combine_vf_wildfire_data
from transformer_modules.attention import MultiHeadAttention

def main(model_flag: str, county_name: str, bAdd_Pop: bool = True, epochs: int = 300, show_plots: bool = True):

  fire_data_path       = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv"
  vf_data_path         = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv"
  pop1_data_path       = "Project/data/cali_county_pop_2000_2010.csv" 
  pop2_data_path       = "Project/data/cali_county_pop_2010_2020.csv"
  start_year           = "2006"
  end_year             = "2015"
  # county_name        = "Tulare"
  split_frac           = .85
  lookback             = 6
  hidden_size          = 64
  num_layers           = 2
  dropout              = 0.2
  learning_rate        = 0.001
  # epochs             = 300
  weight_decay         = 1e-5
  # model_flag           = "transformer"

  # Transformer Properties
  if model_flag == "transformer":
    d_model         = 32
    nheads          = 2
    dim_feedforward = 64
    num_layers      = 1
    dropout         = 0.2

  if bAdd_Pop:
    title_text         = "(Population Added)"
    source_column_labels = ['Fire Incident Count', 'VF Case Count', 'Population']
    df = combine_vf_fire_pop_data(pop1_data_path, pop2_data_path, vf_data_path, fire_data_path, county_name,
                                          start_year, end_year, bInterp=False)
  else:
    title_text         = "(Excl. Population)"
    source_column_labels = ['Fire Incident Count', 'VF Case Count']
    df =  combine_vf_wildfire_data(fire_path=fire_data_path, vf_cases_path=vf_data_path, 
                                   county_name= county_name)

  X_train, y_train, X_test, y_test, scaler = prep_data(df, source_column_labels, 
                                                        split_frac, lookback, b_scaler=True)

  input_size = len(source_column_labels)
  # Able to be swapped out with other models (I hope)
  if model_flag == "lstm":
    model     = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
  elif model_flag == "transformer":
    model = TransformerModel(input_size= input_size, d_model = d_model, nhead= nheads, num_layers = num_layers,
                             dim_feedforward= dim_feedforward, dropout=dropout, attention_impl=MultiHeadAttention)
  else:
    raise NotImplementedError("Other Architectures besides LSTM and Transformer not implemented. " \
                              "If this is a mistake, check your spelling in model_flag variable.")
  criterion = RMSELoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)  

  trainer = Trainer(model = model, criterion = criterion, optimizer = optimizer, scaler = scaler)
  trainer.train(X_train, y_train, X_test, y_test, epochs)

  y_pred, y_true = trainer.evaluate(X_test, y_test)
  if show_plots:
    trainer.visualize_results(y_true, y_pred, county_name, model_type = model_flag, 
                            title_text = title_text, show_plot=True, save_fig = True)
  return y_pred, y_true
if __name__ == "__main__":
  main('lstm', 'Fresno')