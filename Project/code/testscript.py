from data_loader import clean_pop_dataframe

pop_path = 'Project/data/cali_county_pop_2010_2020.csv'
pop1 = clean_pop_dataframe(pop_path)
print(pop1.head())