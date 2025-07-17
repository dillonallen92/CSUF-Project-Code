from dash import html, dcc

def create_layout(df):
    return html.Div([
        html.H1("My First Dashboard"),
        dcc.Dropdown(
            id='continent-dropdown',
            options=[{'label': c, 'value': c} for c in df['continent'].unique()],
            value='Asia'
        ),
        dcc.Graph(id='scatter-plot')
    ])