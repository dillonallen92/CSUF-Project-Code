import configparser

file_path = "Project/configs/masked_lstm_config.ini"
config = configparser.ConfigParser()

config.read(file_path)
lstm_params = config['lstm_params']
for k,v in lstm_params.items():
  print(f"{k} : {v}")


