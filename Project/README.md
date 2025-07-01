# 🌡️ Valley Fever & Wildfire Incidents: Time Series Forecasting with LSTM

This project explores the potential relationship between **wildfire incidents** and **Valley Fever (Coccidioidomycosis)** case rates using a time series modeling pipeline built in **PyTorch**. The goal is to predict future VF case counts using historical trends, with fire data as a possible covariate.

---

## 📁 Project Structure

Project/
├── code/ # All Python source code (models, trainer, utils)
│ ├── main.py # Main pipeline: training, testing, plotting
│ ├── model.py # LSTM (and future model) definitions
│ ├── trainer.py # Trainer class for training/eval/visualization
│ ├── data_utils.py # Data preparation and tensor conversion
│ └── data_loader.py # Loads and fuses VF + fire data per county
├── data/ # Raw datasets
│ ├── CAL_FIRE_Wildland_PublicReport_2000to2018.csv
│ └── coccidioidomycosis_m2000_2015_v0.1.csv
├── media/ # (Optional) Folder for output plots
├── requirements.txt # Python dependencies (optional)
└── README.md # This file


---

## 🚀 Getting Started

1. **Clone the project**:
    ```bash
    git clone https://github.com/your-username/valley-fever-lstm.git
    cd valley-fever-lstm
    ```

2. **Install dependencies**:
    ```bash
    pip install torch numpy pandas scikit-learn matplotlib
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
county        = "Tulare"      # Any valid CA county present in both datasets
lookback      = 12            # Number of months to look back
hidden_size   = 64
num_layers    = 2
dropout       = 0.2
epochs        = 300
