from plotting import make_individual_timeseries, visualize_model_results
from model_interface import create_execute_model
from data_interface import load_combined_data
from dash import Input, Output, callback
import plotly.graph_objects as go 

def register_callbacks(app):
    @app.callback(
        Output('data-table', 'columns'),
        Output('data-table', 'data'),
        Input('county-dropdown', 'value'),
        Input('population-checklist', 'value')
    )
    def update_table(county, pop_option):
        df = load_combined_data(county, use_pop='pop' in pop_option)
        df = df.reset_index()
        columns = [{"name": i, "id": i} for i in df.columns]
        return columns, df.to_dict('records')

    @callback(
    Output('population-plot', 'figure'),
    Output('fire-plot', 'figure'),
    Output('vf-plot', 'figure'),
    Input('county-dropdown', 'value')
    )
    def update_plot(county):
      try:
        print(f"[update_plot] Loading data for county: {county}")
        df = load_combined_data(county, use_pop=True)
        print(f"[update_plot] DF shape: {df.shape}")
        print(df.head())

        plots = make_individual_timeseries(df)
        
        return (
           plots.get('population'),
           plots.get('fire'),
           plots.get('vf')
        )
      except Exception as e:
        print(f"[update_plot] ERROR: {e}")
        import plotly.graph_objects as go
        error_fig = go.Figure().add_annotation(
            text=f"Plotting error: {str(e)}",
            showarrow=False
        )
        return error_fig, error_fig, error_fig
    
    @callback(
          Output('model-summary-plot', 'figure'),
          Input('select-model-dropdown', 'value'),
          Input('county-dropdown', 'value'),
          Input('population-checklist', 'value')
    )
    def update_model_summary_plot(model_flag, county, pop_option):
      df             = load_combined_data(county, use_pop='pop' in pop_option )
      y_pred, y_true = create_execute_model(df = df, model_flag = model_flag, use_pop='pop' in pop_option)
      model_plot     = visualize_model_results(y_pred, y_true, county, model_flag)
      return model_plot
