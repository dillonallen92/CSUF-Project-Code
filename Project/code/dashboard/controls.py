from dash import dcc 

counties = ['Fresno', 'Kern', 'Kings', 'Los Angeles', 'Tulare']
models   = ['LSTM', 'MultiHeadAttention Transformer']

county_dropdown = dcc.Dropdown(
  id="county-dropdown",
  options = [{'label' : county, 'value': county} for county in counties],
  value = 'Fresno',
  style = {'width' : '50%'}
)

population_checklist = dcc.Checklist(
  id="population-checklist",
  options = [{'label' : 'Include Population', 'value':'pop'}],
  value = [],
  labelStyle={'display' : 'inline-block', 'margin-right':'15px'}
)

pop_option_radio = dcc.RadioItems(
  id = 'popOption-radio',
  options=[
    {'label': 'Population Copy', 'value': 'popcopy'},
    {'label': 'Linearly Interpolation', 'value': 'poplininterp'}
  ],
  value = None,
  labelStyle={'display':'inline-block', 'margin-right':'15px'},
  style = {'display':'none'}
)

vf_case_mode_radio = dcc.RadioItems(
  id='vfCaseMode-radio',
  options=[
    {'label': 'Case Count', 'value':'casecount'},
    {'label': 'Case Rate', 'value':'caserate'}
  ],
  value='casecount',
  labelStyle={'display': 'inline-block', 'margin-right':'15px'}
)

plot_county_dropdown = dcc.Dropdown(
  id='plot-county-dropdown', 
  options = [{'label':county, 'value': county} for county in counties],
  value = 'Fresno',
  style = {'width' : '50%'}
)

select_model_dropdown = dcc.Dropdown(
  id="select-model-dropdown",
  options = [{'label': model, 'value' : model } for model in models],
  value = 'LSTM',
  style = {'width' : '50%'}
)