# this will be the plotly app to launch the dashboard

import dash 
from dash import dcc, html 
import plotly.express as px 
import pandas as pd

# Sample data
df = px.data.gapminder().query("year == 2007")

# Create app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("My First Dashboard"),
    dcc.Dropdown(
        id='continent-dropdown',
        options=[{'label': c, 'value': c} for c in df['continent'].unique()],
        value='Asia'
    ),
    dcc.Graph(id='scatter-plot')
])

# Callbacks
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('continent-dropdown', 'value')]
)
def update_graph(selected_continent):
    filtered_df = df[df['continent'] == selected_continent]
    fig = px.scatter(filtered_df, x='gdpPercap', y='lifeExp',
                     size='pop', color='country', hover_name='country',
                     log_x=True)
    return fig

# Run server
if __name__ == '__main__':
    app.run(debug=True)