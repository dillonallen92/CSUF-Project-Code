import plotly.express as px
import pandas as pd

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
