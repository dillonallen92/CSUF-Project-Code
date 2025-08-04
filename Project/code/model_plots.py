import matplotlib.pyplot as plt 
from main import main

def model_comparison_subplots_by_count(county_name: str):

  model_flag_arr = ["lstm", "lstm", "transformer", "transformer"]
  bAdd_Pop_Arr   = [False, True, False, True]

  fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = True, sharey = True)

  for plot_idx in range(4):
    row = plot_idx //  2 
    col = plot_idx %  2 
    ax  = axes[row, col]

    y_pred, y_true = main(model_flag_arr[plot_idx], county_name= county_name, bAdd_Pop= bAdd_Pop_Arr[plot_idx], show_plots=False)

    ax.plot(y_pred, label = 'Predicted', linestyle = '--', marker = 'o')
    ax.plot(y_true, label = 'Truth', linestyle = '-', marker = 'o')
    ax.set_title(f"{model_flag_arr[plot_idx].upper()} | Pop : {bAdd_Pop_Arr[plot_idx]}")
    ax.set_xlabel("Months")
    ax.set_ylabel("VF Case Counts")
    ax.legend()

  plt.suptitle(f"Valley Fever Model Forecast Comparison for {county_name}")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  model_comparison_subplots_by_count("Fresno")

