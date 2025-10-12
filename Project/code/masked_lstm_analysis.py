import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from loss_functions import RMSELoss
from datetime import datetime

def read_data(file_path:str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(file_path)
    return data
  
def format_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # turn the Year-Month column into the index
    df: pd.DataFrame = df.set_index('Year-Month')
    # convert the index to a datetime object
    df.index = pd.to_datetime(df.index)
    return df

def create_feature_and_target_arrays(df: pd.DataFrame, 
                                     target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols : list[str] = [col for col in df.columns if col != target_col]
    X : pd.DataFrame = df[feature_cols]
    y : np.ndarray = df[target_col].to_numpy()
    return X, y

def generate_padded_data(feature_df: pd.DataFrame, 
                         df_window_sizes: pd.DataFrame, 
                         target_vec:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data_cols : list[str] = feature_df.columns.tolist()
    num_features : int = len(data_cols)
    window_sizes : dict[str, int] = df_window_sizes.set_index('feature')['window_size'].to_dict()
    max_window_size : int = max(window_sizes.values())
    num_samples : int = feature_df.shape[0] - max_window_size + 1; # how many feature vectors I will have as an np.ndarray
    padded_data : np.ndarray = np.zeros((max_window_size, num_features, num_samples))
    
    # Now I need to fill in padded data... For each row in the df_data dataframe, I need to 
    # fill the padded_data array with the appropriate feature vector size. For example, some of the 
    # features have a window size of 12, so I will take all 12 data points. In other features, if the
    # window size is 1, I will have 11 zeros and 1 datapoint at the end. 
    
    
    for i in range(num_samples):
        for j, feature in enumerate(data_cols):
            window_size:int = window_sizes[feature]
            # Fill in the last 'window_size' entries with actual data
            padded_data[max_window_size - window_size:, j, i] = feature_df[feature].iloc[i:i + window_size].to_numpy()
            # The rest are already zeros due to initialization
    
    y_tgt_adj: np.ndarray = target_vec[(max_window_size-1):]
    return padded_data, y_tgt_adj
    
def create_masking_vector(feature_df: pd.DataFrame, df_window_sizes:pd.DataFrame) -> np.ndarray:
    window_sizes_filtered : pd.DataFrame = df_window_sizes[df_window_sizes['feature'] != 'All Features']
    window_sizes: dict[str, int] = window_sizes_filtered.set_index('feature')['window_size'].to_dict()
    max_window_size : int = max(window_sizes.values())
    num_features : int = window_sizes_filtered.shape[0]
    masking_matrix: np.ndarray = np.zeros((max_window_size, num_features))
    data_cols: list[str] = feature_df.columns.tolist()
    # now that we have the masking matrix of zeros, I need to replace each of the columns 
    # with the necessary 1's based off the window length

    for j, feature in enumerate(data_cols):
        # print(f"index {j} relates to feature {feature}")
        window_size : int = window_sizes[feature]
        if window_size == max_window_size:
            masking_matrix[:,j] = np.ones((max_window_size, ))
        else:
            masking_matrix[(max_window_size - window_size):, j] = np.ones((window_size, ))

    # print(masking_matrix)
    return masking_matrix


class MaskedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # PyTorch applies dropout only when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:     (batch, seq_len, input_size)
        mask:  (batch, seq_len, input_size) entries of 0/1 indicating which inputs are valid.
        """
        masked_x = x * mask
        outputs, _ = self.lstm(masked_x)
        outputs = self.dropout(outputs)

        valid_steps = (mask.sum(dim=-1) > 0).float()
        pooled = (outputs * valid_steps.unsqueeze(-1)).sum(dim=1)
        denom = valid_steps.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = pooled / denom
        return self.regressor(pooled).squeeze(-1)


def to_tensors(padded_data: np.ndarray, 
               targets: np.ndarray, 
               mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts numpy arrays to torch tensors and aligns dimensions for model consumption.
    """
    # padded_data comes in as (seq_len, num_features, num_samples); transpose to (samples, seq_len, num_features).
    tensor_data = torch.from_numpy(np.transpose(padded_data, (2, 0, 1))).float()
    tensor_targets = torch.from_numpy(targets).float()
    tensor_mask = torch.from_numpy(mask).float()
    return tensor_data, tensor_targets, tensor_mask


def build_dataloaders(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_frac: float = 0.8,
    batch_size: int = 32,) -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(features, targets)
    train_size = int(len(dataset) * train_frac)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_masked_lstm(
    model: MaskedLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mask: torch.Tensor,
    epochs: int = 50,
    learning_rate: float = 1e-3,) -> MaskedLSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mask = mask.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            mask_batch = mask.unsqueeze(0).expand(batch_inputs.size(0), -1, -1)

            predictions = model(batch_inputs, mask_batch)
            loss = criterion(predictions, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                mask_batch = mask.unsqueeze(0).expand(batch_inputs.size(0), -1, -1)

                predictions = model(batch_inputs, mask_batch)
                loss = criterion(predictions, batch_targets)
                val_loss += loss.item() * batch_inputs.size(0)

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

if __name__ == "__main__":
    ###################
    # Adjustable Data #
    ###################
    
    county_name = "Kern"
    df_agg: pd.DataFrame = read_data(f"Project/data/{county_name}_Aggregate.csv")
    best_window_vals: pd.DataFrame = read_data(f"Project/data/{county_name.lower()}_lstm_best_features_window_results.csv")
    learning_rate = 1e-2
    epochs = 200
    train_frac = 0.8
    hidden_size = 64
    save_fig = True 
    ################
    # Do Not Touch #
    ################
    
    df_data: pd.DataFrame = format_feature_dataframe(df_agg)
    print(f" ---- {county_name} Aggregate Data ---- ")
    print(df_data)
    print(" ---- Best Window Values ---- ")
    print(best_window_vals)
    
    df_features, target_vec = create_feature_and_target_arrays(df_data, target_col='VFRate')
    print(" ---- Feature DataFrame ---- ")
    print(df_features)
    print(" ---- Target Vector ---- ")
    print(target_vec)
    
    padded_data, tgt_adj = generate_padded_data(df_features, best_window_vals, target_vec=target_vec)
    print(" ---- Padded Data Shape ---- ")
    print(padded_data.shape)

    print(" --- Padded Data Example --- ")
    # Print the first sample's padded data, put into a dataframe to be easier to read in console
    print(pd.DataFrame(padded_data[:,:,0]))    
    print(" --- Corresponding Target Value --- ")
    print(target_vec[11])  # Print the target value corresponding to the first sample
    print("--- Adjusted Target Vec Value ---")
    print(tgt_adj[0])    

    print("--- Masking Matrix ---")
    masking_matrix: np.ndarray = create_masking_vector(df_features, best_window_vals)
    print(masking_matrix)

    feature_tensor, target_tensor, mask_tensor = to_tensors(padded_data, tgt_adj, masking_matrix)
    train_loader, val_loader = build_dataloaders(feature_tensor, target_tensor, 
                                                 train_frac=train_frac, batch_size=16)

    masked_lstm = MaskedLSTM(input_size=feature_tensor.size(-1), hidden_size=hidden_size, num_layers=2, dropout=0.2)
    masked_lstm = train_masked_lstm(
        model=masked_lstm,
        train_loader=train_loader,
        val_loader=val_loader,
        mask=mask_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masked_lstm.eval()
    with torch.no_grad():
        expanded_mask = mask_tensor.unsqueeze(0).expand(feature_tensor.size(0), -1, -1).to(device)
        predictions = masked_lstm(feature_tensor.to(device), expanded_mask).cpu().numpy()
   
    split_idx = np.floor(len(predictions) * train_frac)
    plt.plot(np.arange(1, len(tgt_adj) + 1), tgt_adj, label="VF Rate (Actual)", linestyle="-.")
    plt.plot(np.arange(1, len(predictions)+1), predictions, label="VF Rate (Predicted)", linestyle="-.")
    plt.axvline(x = split_idx, color='r', linestyle='--')
    plt.xlabel("Months")
    plt.ylabel("VF Case Rate")
    plt.title(f"{county_name} LSTM VF Case Rate (Ind. Sliding Window) \nlr = {learning_rate}, hl = {hidden_size}, train/test = {train_frac}|{1-train_frac}")
    plt.grid()
    plt.legend()
    plt.show()    
    
    if save_fig:
        print(" ---- Saving Image ----")
        plt.figure(figsize=(12, 6))
        split_idx = np.floor(len(predictions) * train_frac)
        plt.plot(np.arange(1, len(tgt_adj) + 1), tgt_adj, label="VF Rate (Actual)", linestyle="-.")
        plt.plot(np.arange(1, len(predictions)+1), predictions, label="VF Rate (Predicted)", linestyle="-.")
        plt.axvline(x = split_idx, color='r', linestyle='--')
        plt.xlabel("Months")
        plt.ylabel("VF Case Rate")
        plt.title(f"{county_name} LSTM VF Case Rate (Ind. Sliding Window) \nlr = {learning_rate}, hl = {hidden_size}, train/test = {train_frac}|{1-train_frac}")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        img_str = f"Project/plots/lstm/{county_name}_VF_Sliding_Window_plot_{datetime.today()}.png"
        plt.savefig(img_str)
        print(f"Image saved to: {img_str}")
    print("---- Analysis Complete ----")
    
    
