from controls import county_dropdown, population_checklist, \
  plot_county_dropdown, select_model_dropdown, pop_option_radio

from dash import html, dcc, dash_table

layout = html.Div([
  html.Div([
    html.H1("Valley Fever Project Dashboard")
  ]),
  html.Div([
  html.Label("Select County: "),
  html.Div([
  county_dropdown], style = {'marginLeft': '15px', 'width': '800px'}), html.Div([])], style={'display':'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
  html.Div([
    population_checklist,
    pop_option_radio], style={'marginBottom': '15px'}),
  dcc.Tabs([
    dcc.Tab(label='Explore Data Table', children = [
      # population_checklist,
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
      html.H1("Model Selection and Results"),
      html.Div([
        select_model_dropdown
      ], style = {'marginTop' : '15px', 'marginBottom' : '15px'}),
      dcc.Graph(id='model-summary-plot')
    ])
  ])
])