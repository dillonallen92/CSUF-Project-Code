from dash import Input, Output
from components.charts import make_scatter_plot

def register_callbacks(app, df):
    @app.callback(Output('scatter-plot', 'figure'), Input('continent-dropdown', 'value'))
    def update_graph(selected_continent):
        filtered_df = df[df['continent'] == selected_continent]
        return make_scatter_plot(filtered_df)