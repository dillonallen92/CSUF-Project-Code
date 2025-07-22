import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np 

def make_individual_timeseries(df: pd.DataFrame):
    df = df.copy().reset_index()

    # Rename index column to 'Date' if needed
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    elif df.columns[0] != 'Date':
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    plots = {}

    # Plot 1: Population (if present)
    if 'Population' in df.columns:
        plots['population'] = px.scatter(df, x='Date', y='Population', title='Population Over Time')

    # Plot 2: Fire Incident Count
    if 'Fire Incident Count' in df.columns:
        plots['fire'] = px.scatter(df, x='Date', y='Fire Incident Count', title='Fire Incidents Over Time')

    # Plot 3: VF Case Count
    if 'VF Case Count' in df.columns:
        plots['vf'] = px.scatter(df, x='Date', y='VF Case Count', title='VF Cases Over Time')

    return plots

def visualize_model_results(y_pred, y_test, county, model_flag):
    fig = go.Figure()
    num_months = np.arange(0, len(y_pred), 1)
    fig.add_trace(go.Scatter(x = num_months, y = y_pred, mode='markers', name='Predicted Value'))
    fig.add_trace(go.Scatter(x = num_months, y = y_test, mode = "markers", name = "True Value"))

    fig.update_layout(
        title=f"Model: {model_flag} | County: {county}",
        xaxis_title="Month",
        yaxis_title="Value",
        legend_title="Legend",
    )
    
    return fig

