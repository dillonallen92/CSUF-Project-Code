from controls import county_dropdown, population_checklist, plot_county_dropdown

from dash import html, dcc, dash_table

layout = html.Div([
  dcc.Tabs([
    dcc.Tab(label='Explore Data Table', children = [
      county_dropdown, 
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
      plot_county_dropdown,
      dcc.Graph(id='population-plot'),
      dcc.Graph(id='fire-plot'),
      dcc.Graph(id='vf-plot')
    ])
  ])
])