import sys
from pathlib import Path 
import pandas as pd
import torch
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import LSTM, TransformerModel
from trainer import Trainer 
from data_utils import prep_data
from loss_functions import RMSELoss
from transformer_modules.attention import MultiHeadAttention

# default parameters right now, will allow users to change later
# probably
def create_execute_model(df: pd.DataFrame, model_flag: str, use_pop: bool) -> tuple[torch.tensor, torch.tensor]:
  
  if use_pop:
    source_column_labels = ['Fire Incident Count', 'VF Case Count', 'Population']
  else:
    source_column_labels = ['Fire Incidenct Count', 'VF Case Count']
  
  # Training and model parameters (general)
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

  # Transformer Properties if model_flag is transformer
  if model_flag == "MultiHeadAttention Transformer":
    d_model         = 32
    nheads          = 2
    dim_feedforward = 64
    num_layers      = 1
    dropout         = 0.2
  
  X_train, y_train, X_test, y_test, scaler = prep_data(df, source_column_labels, 
                                                       split_frac= split_frac,
                                                        lookback = lookback,
                                                         b_scaler = True)
  
  if model_flag == "LSTM":
    model = LSTM(input_size, hidden_size, dropout, num_layers)
  elif model_flag == "MultiHeadAttention Transformer":
    model = TransformerModel(input_size, d_model, nheads, num_layers, dim_feedforward,
                             dropout, attention_impl=MultiHeadAttention)
  else:
    raise ValueError("Not a valid MODEL_FLAG parameter")
  
  criterion = RMSELoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate,
                         weight_decay = weight_decay)
  
  trainer = Trainer(model = model, criterion = criterion,
                    optimizer = optimizer, scaler = scaler)
  
  trainer.train(X_train, y_train, X_test, y_test, epochs)
  
  y_pred, y_test = trainer.evaluate(X_test, y_test)
  return y_pred, y_test
  
