from controls import county_dropdown, population_checklist, plot_county_dropdown

from dash import html, dcc, dash_table

layout = html.Div([
  html.Div([
  html.Label("Select County: "),
  html.Div([
  county_dropdown], style = {'marginLeft': '15px', 'width': '800px'})], style={'display':'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
  dcc.Tabs([
    dcc.Tab(label='Explore Data Table', children = [
      population_checklist,
      html.Br(),
      dash_table.DataTable(
          id='data-table',
          columns=[],  # initially empty
          data=[],     # initially empty
          page_size=20,
          style_table={
              'overflowX': 'auto',
              'maxHeight': '500px',
              'overflowY': 'scroll'
          },
          style_cell={
              'textAlign': 'left',
              'padding': '5px',
              'minWidth': '100px',
              'width': '100px'
          },
      )
    ]),
    dcc.Tab(label='Explore Data Plots', children=[
      dcc.Graph(id='population-plot'),
      dcc.Graph(id='fire-plot'),
      dcc.Graph(id='vf-plot')
    ]),
    dcc.Tab(label = "Model Prediction", children = [
      html.H1("Hello World")
    ])
  ])
])