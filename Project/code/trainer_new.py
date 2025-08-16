import torch 
import numpy as np
import matplotlib.pyplot as plt 
from datetime import date

# Creating a Trainer Class to contain Training/Testing/Visualizing
class TrainerNewNew:
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
    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        self.model.train()
        output = self.model(X_train)
        loss = self.criterion(output, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        if epoch % 10 == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(X_test), y_test)
                history['test_loss'].append(val_loss.item())
                print(f"Epoch {epoch+1}/{epochs} - Training Loss {loss.item():.4f}, Testing Loss {val_loss.item():.4f}")
    
    # Capture the final training predictions
    self.model.eval()
    with torch.no_grad():
        final_train_preds = self.model(X_train).detach().cpu().numpy()
        
    # Also inverse transform the training data for later plotting
    y_train_true = self.scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
    final_train_preds_inv = self.scaler.inverse_transform(final_train_preds).flatten()
    
    return history, final_train_preds_inv, y_train_true
    
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
        preds = self.model(X_test).detach().cpu().numpy()

        # Inverse transform the predictions using the y_scaler
        # The y_scaler was fit on a 1-D array, so the predictions should be reshaped
        vf_pred = self.scaler.inverse_transform(preds)

        # Inverse transform the true values using the y_scaler
        # The y_scaler was fit on a 1-D array, so the true values should be reshaped
        vf_true = self.scaler.inverse_transform(y_test.cpu().numpy())
        
    return vf_pred.flatten(), vf_true.flatten()
  
  def visualize_results(self, true, pred, county_name="", model_type = "LSTM", title_text = "", show_plot = True, save_fig = False):
    """
    Function to visualize the prediction vs true (test) vector

    Inputs:
      - True: True data (test or validation target vector)
      - Pred: Prediction data from the model evaluation function
    """
    if show_plot:
      plt.figure(figsize=(12, 6))
      plt.plot(true, label="True Values")
      plt.plot(pred[1:], label = "Predicted Values", linestyle="--")
      plt.title(f"{county_name} {model_type} {title_text} True vs Predicted Valley Fever Case Rates")
      plt.xlabel("Months")
      plt.ylabel("Case Rates")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()
    
    if save_fig:
      plt.figure(figsize=(12, 6))
      plt.plot(true, label="True Values")
      plt.plot(pred[1:], label = "Predicted Values", linestyle="--")
      plt.title(f"{county_name} {model_type} {title_text} LSTM True vs Predicted Valley Fever Case Rates")
      plt.xlabel("Months")
      plt.ylabel("Case Rates")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      img_str = f"Project/plots/{model_type}/{county_name}_{title_text}_plot_{date.today()}.png"
      plt.savefig(img_str)