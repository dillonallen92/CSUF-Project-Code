import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 

def read_data(file_path:str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(file_path)
    return data




if __name__ == "__main__":
    
    df_fresno_agg = read_data("Project/data/Fresno_Aggregate.csv")
    best_window_vals = read_data("Project/data/fresno_lstm_best_feature_window_results.csv")
    print(" --- Fresno Aggregate Data --- ")
    print(df_fresno_agg)
    print(" --- Best Window Values --- ")
    print(best_window_vals)