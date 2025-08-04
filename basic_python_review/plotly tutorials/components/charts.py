import plotly.express as px

def make_scatter_plot(df):
    return px.scatter(df, x='gdpPercap', y='lifeExp',
                      size='pop', color='country', hover_name='country',
                      log_x=True)