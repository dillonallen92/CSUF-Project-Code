import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.utils.data as data 
import torch.optim as optim

# Generate synthetic data
t = np.linspace(0, 100, 1000)
y = np.sin(t) + 0.1*np.random.randn(len(t))

# Classes for the architecture
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len = 5000):
    super().__init__()
    pe = torch.zeros(max_len, d_model) # this creates the encoding matrix (max_len, d_model) 
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log10(10000.0)/d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe) # loads into to model and will not be updated on training

  def forward(self, x):
    x = x + self.pe[:, :x.size(1), :] # positional encoding update, tensor of form (batch, seq_len, d_model)
    return x
  
class TransformerSineDenoiser(nn.Module):
  def __init__(self, input_dim = 1, d_model = 64, nhead = 4, num_layers = 2, 
               dim_feedforward = 128):
    super().__init__()
    self.input_proj = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead = nhead, dim_feedforward=dim_feedforward)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.output_proj = nn.Linear(d_model,1)
  
  def forward(self, src):
    # src is the encoder vector, conventially used instead of x
    src = self.input_proj(src)
    src = self.pos_encoder(src)
    src = src.permute(1, 0, 2)
    out = self.transformer_encoder(src)
    out = out.permute(1, 0, 2) # undo the permutation above
    return self.output_proj(out)

def create_sequences(t, y, seq_len = 50):
  inputs, targets = [], [] 
  for i in range(len(y) - seq_len):
    noisy_seq = y[i:i+seq_len]
    clean_seq = np.sin(t[i:i+seq_len])
    inputs.append(noisy_seq)
    targets.append(clean_seq)
  return np.array(inputs), np.array(targets)

# Split 70/30
split_idx = int(0.7 * len(y))
t_train, y_train = t[:split_idx], y[:split_idx]
t_test,  y_test  = t[split_idx:], y[split_idx:]

seq_len = 200 
X_train, Y_train = create_sequences(t_train, y_train, seq_len=seq_len)

# Tensors and dataloader
X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)

dataset = data.TensorDataset(X_tensor, Y_tensor)
loader  = data.DataLoader(dataset, batch_size = 32, shuffle = True)

# Create an instance of the model so we can train and test
model = TransformerSineDenoiser()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training Loop
epochs = 40 

for epoch in range(epochs):
  model.train()
  total_loss = 0

  for xb, yb in loader:
    preds = model(xb)
    loss  = loss_fn(preds, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  
  if epoch % 10 == 0:
    print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss/len(loader):.6f}")

# Time to predict and see how it does
# Predict a denoised sequence from test set
model.eval()
with torch.no_grad():
  test_seq = y_test[:seq_len]  # noisy test sequence
  test_t   = t_test[:seq_len]  # corresponding clean time
  clean_gt = np.sin(test_t)    # ground truth sine

  input_tensor = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
  prediction = model(input_tensor).squeeze().numpy()

# Plot
plt.plot(test_seq, label='Noisy Input', linestyle='dotted')
plt.plot(clean_gt, label='True Clean')
plt.plot(prediction, label='Predicted Denoised')
plt.legend()
plt.title("Transformer Denoising on 30% Unseen Test Sequence")
plt.show()