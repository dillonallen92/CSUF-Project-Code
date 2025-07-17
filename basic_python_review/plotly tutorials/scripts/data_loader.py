import plotly.express as px 

def load_gapminder_2007():
  df = px.data.gapminder().query("year == 2007")
  return df 
