import torch 
import numpy as np
import matplotlib.pyplot as plt 

# Creating a Trainer Class to contain Training/Testing/Visualizing
class Trainer:
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
    """
    Training Loop Function

    Inputs:
      - X_train: Training matrix X
      - y_train: training target vector y
      - X_test: Test matrix X
      - y_test: Target vector y
      - epochs: Number of Epochs to train
    """
    for epoch in range(epochs):
      self.model.train()
      output = self.model(X_train)
      loss   = self.criterion(output, y_train)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if epoch % 10 == 0:
        self.model.eval()
        val_loss = self.criterion(self.model(X_test), y_test)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss {loss.item():.4f}, Testing Loss {val_loss.item():.4f}")
    
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
        preds = self.model(X_test).detach().numpy()

        dummy = np.zeros((len(preds), 2))
        dummy[:, 1] = preds[:, 0]
        vf_pred = self.scaler.inverse_transform(dummy)[:,1]

        dummy[:,1] = y_test.numpy().flatten()
        vf_true = self.scaler.inverse_transform(dummy)[:,1]
      
      return vf_pred, vf_true
  
  def visualize_results(self, true, pred):
    """
    Function to visualize the prediction vs true (test) vector

    Inputs:
      - True: True data (test or validation target vector)
      - Pred: Prediction data from the model evaluation function
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True Values")
    plt.plot(pred, label = "Predicted Values", linestyle="--")
    plt.title("LSTM True vs Predicted Valley Fever Case Rates")
    plt.xlabel("Months")
    plt.ylabel("Case Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()