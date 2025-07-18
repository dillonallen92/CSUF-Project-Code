import dash
from scripts.data_loader import load_gapminder_2007
from components.layout import create_layout
from callbacks.update_charts import register_callbacks

# Load data
df = load_gapminder_2007()

# Initialize Dash app
app = dash.Dash(__name__)

# Set layout
app.layout = create_layout(df)

# Register callbacks
register_callbacks(app, df)

# Run server
if __name__ == '__main__':
    app.run(debug=True)