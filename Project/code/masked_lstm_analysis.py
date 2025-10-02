import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 

def read_data(file_path:str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(file_path)
    return data
  
def format_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # turn the Year-Month column into the index
    df = df.set_index('Year-Month')
    # convert the index to a datetime object
    df.index = pd.to_datetime(df.index)
    return df

def create_feature_and_target_arrays(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols : list[str] = [col for col in df.columns if col != target_col]
    X : pd.DataFrame = df[feature_cols]
    y : np.ndarray = df[target_col].to_numpy()
    return X, y

def generate_padded_data(feature_df: pd.DataFrame, df_window_sizes: pd.DataFrame) -> np.ndarray:
    data_cols : list[str] = feature_df.columns.tolist()
    num_features : int = len(data_cols)
    window_sizes : dict[str, int] = df_window_sizes.set_index('feature')['window_size'].to_dict()
    max_window_size : int = max(window_sizes.values())
    num_samples : int = feature_df.shape[0] - max_window_size + 1; # how many feature vectors I will have as an np.ndarray
    padded_data : np.ndarray = np.zeros((max_window_size, num_features, num_samples))
    
    # Now I need to fill in padded data... For each row in the df_data dataframe, I need to 
    # fill the padded_data array with the appropriate feature vector size. For example, some of the 
    # features have a window size of 12, so I will take all 12 data points. In other features, the
    # window size is 1, so I will have 1 data point and 11 zeroes. 
    
    
    
    return padded_data
    
  


if __name__ == "__main__":
    
    df_fresno_agg = read_data("Project/data/Fresno_Aggregate.csv")
    best_window_vals = read_data("Project/data/fresno_lstm_best_feature_window_results.csv")
    df_fresno = format_feature_dataframe(df_fresno_agg)
    print(" ---- Fresno Aggregate Data ---- ")
    print(df_fresno)
    print(" ---- Best Window Values ---- ")
    print(best_window_vals)
    
    [df_features, target_vec] = create_feature_and_target_arrays(df_fresno, target_col='VFRate')
    print(" ---- Feature DataFrame ---- ")
    print(df_features)
    print(" ---- Target Vector ---- ")
    print(target_vec)
    
    padded_data = generate_padded_data(df_features, best_window_vals)
    print(" ---- Padded Data Shape ---- ")
    print(padded_data.shape)
    