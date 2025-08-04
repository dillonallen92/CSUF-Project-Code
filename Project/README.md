# 🌡️ Valley Fever & Wildfire Incidents: Time Series Forecasting with LSTM and Transformer Models

This project explores the potential relationship between **wildfire incidents** and **Valley Fever (Coccidioidomycosis)** case rates using a time series modeling pipeline built in **PyTorch**. The goal is to predict future VF case counts using historical trends. The models used in this codebase will be focused around LSTM models and Transformer models like Vanilla Transformer, Informer, Temporal-Fusion Transformer, and potential complementary models like DLinear.

---

## 🚀 Getting Started

1. **Clone the project**:

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the model**:
    ```bash
    cd code
    python main.py
    ```

---

## ⚙️ Configuration Options

Edit `main.py` to set modeling options and pick the county:

```python
county_name   = "Tulare"      # Any valid CA county present in both datasets
lookback      = 12            # Number of months to look back
hidden_size   = 64
num_layers    = 2
dropout       = 0.2
epochs        = 300
