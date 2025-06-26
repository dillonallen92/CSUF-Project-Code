import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch
import torch.nn as nn 
import torch.optim as optim 

# Flags for certain things to run
DIAGNOSTIC_PLOT = 0

# create the artificial data
t = np.linspace(0, 100, 100000)
y = np.sin(t) + 0.1*np.random.randn(len(t))
y_clean = np.sin(t)

# plot the artificial data for now
if DIAGNOSTIC_PLOT:
  plt.plot(t, y)
  plt.show()

def create_dataset(series, lookback = 200):
  '''
  This function is responsible for creating the tensors that will be used in the LSTM
  from the dataset.
  '''

  X, y = [], [] 
  for i in range(len(series) - lookback):
    X.append(series[i:i+lookback])
    y.append(series[i+lookback])
  
  return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_frac = .70
train_idx = int(len(y)*train_frac)
X_train, y_train = create_dataset(y[:train_idx], lookback=50)
X_test, y_test   = create_dataset(y[train_idx:], lookback = 50)
y_clean_test = y_clean[train_idx:] 
print(f"Train Shape: X: {X_train.shape}, y: {y_train.shape}\nTest Shape: X: {X_test.shape}, y: {y_test.shape}")

# Create SineAnalyzer class that has the LSTM architecture
class SineAnalyzer(nn.Module):
  
  def __init__(self, input_size, hidden_size, num_layers):
    super().__init__()
    self.lstm = nn.LSTM(input_size= input_size,
                        hidden_size= hidden_size,
                        num_layers= num_layers,
                        batch_first= True)
    self.linear = nn.Linear(hidden_size, input_size)
  
  def forward(self, x):
    out, _ = self.lstm(x[:,-1,:])
    out    = self.linear(out)
    return out 

# create an instance of the lstm and train/predict
input_size = 1
hidden_size = 64 
num_layers = 1
model = SineAnalyzer(input_size=input_size, 
                     hidden_size=hidden_size,
                     num_layers=num_layers)

# Create the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modify the data to be digestible for the LSTM
# Training:
X_train = X_train.unsqueeze(-1)
y_train = y_train.unsqueeze(-1)

# Testing:
X_test = X_test.unsqueeze(-1)
y_test = y_test.unsqueeze(-1)

# Training Loop
num_epochs = 100

for epoch in range(num_epochs):
  model.train()
  output = model(X_train)
  loss   = criterion(output, y_train)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    preds = model(X_test)

# Flatten predictions and true values
# Convert tensors to lists for plotting
y_train_plot = y_train.squeeze().cpu().tolist()
y_test_plot = y_test.squeeze().cpu().tolist()
preds_plot = preds.squeeze().cpu().tolist()

# Join train and test series to align them on the x-axis
true_full = y_train_plot + y_test_plot
pred_full = [None] * len(y_train_plot) + preds_plot

# Plot true values (solid for train, dotted for test)
plt.plot(range(len(y_train_plot)), y_train_plot, label="Noisy Sine (train)", linestyle="-")
plt.plot(range(len(y_train_plot), len(true_full)), y_clean_test[:len(preds_plot)], label="True (test, clean)", linestyle="--")

# Plot predicted values (dotted only for test portion)
plt.plot(range(len(pred_full)), pred_full, label="Predicted (test, clean)", linestyle="--")

plt.title("One-Step Prediction of Sine Wave")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
