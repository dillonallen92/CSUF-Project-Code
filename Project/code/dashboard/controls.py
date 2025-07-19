from dash import dcc 

counties = ['Fresno', 'Kern', 'Kings', 'Los Angeles', 'Tulare']

county_dropdown = dcc.Dropdown(
  id="county-dropdown",
  options = [{'label' : county, 'value': county} for county in counties],
  value = 'Fresno',
  style = {'width' : '50%'}
)

population_checklist = dcc.Checklist(
  id="population-checklist",
  options = [{'label' : 'Include Population', 'value':'pop'}],
  value = ['pop'],
  labelStyle={'display' : 'inline-block', 'margin-right':'15px'}
)

plot_county_dropdown = dcc.Dropdown(
  id='plot-county-dropdown', 
  options = [{'label':county, 'value': county} for county in counties],
  value = 'Fresno',
  style = {'width' : '50%'}
)