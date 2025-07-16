import pandas as pd 

def combine_vf_wildfire_data(fire_path: str, vf_cases_path: str, county_name: str) -> pd.DataFrame:
    """
    Loads and returns a combined monthly time series of Valley Fever cases and Fire Incidents
    for the specified county from 2006-01 to 2015-12.
    """

    fire_data = pd.read_csv(fire_path, low_memory=False)
    vf_data = pd.read_csv(vf_cases_path, low_memory=False)

    # --- Filter County ---
    fire_county = fire_data[fire_data['County'] == county_name].copy()
    vf_county = vf_data[vf_data['County'] == county_name].copy()

    # --- Process Fire Data ---
    fire_county["Incident Date"] = pd.to_datetime(fire_county["Incident Date"], errors="coerce")
    fire_county = fire_county[
        (fire_county["Incident Date"] >= "2006-01-01") & (fire_county["Incident Date"] <= "2018-12-31")
    ].set_index("Incident Date")

    # Monthly resampling and zero-fill
    fire_monthly = fire_county.resample("M").size()
    all_months = pd.date_range("2006-01-01", "2018-12-31", freq="M")
    fire_monthly = fire_monthly.reindex(all_months, fill_value=0)

    fire_df = fire_monthly.reset_index()
    fire_df.columns = ["Month", "Fire Incident Count"]
    fire_df["Month"] = fire_df["Month"].dt.to_period("M").astype(str)

    # --- Process Valley Fever Data ---
    vf_trimmed = vf_county.drop(columns=["State", "County", "FIPS"], errors="ignore")
    vf_long = vf_trimmed.T.reset_index()
    vf_long.columns = ["Month", "VF Case Count"]
    vf_long["Month"] = pd.to_datetime(vf_long["Month"], format="%Y/%m", errors="coerce")
    vf_long["Month"] = vf_long["Month"].dt.to_period("M").astype(str)

    # --- Trim Date Range for Both ---
    fire_trim = fire_df[(fire_df["Month"] >= "2006-01") & (fire_df["Month"] <= "2015-12")]
    vf_trim = vf_long[(vf_long["Month"] >= "2006-01") & (vf_long["Month"] <= "2015-12")]

    # --- Merge Datasets ---
    combined = pd.merge(fire_trim, vf_trim, on="Month", how="inner")

    return combined

# Population Data Cleaning Functions
def clean_pop_dataframe(pop_data_path: str) -> pd.DataFrame:
    """
    Clean_Pop_Dataframe: This function takes in a csv and cleans the
    dataframe (removing ', California' and '.' before county name)
    and prepares the dataframe for potential inner joins with other population
    dataframes (from other years)

    Input: 
        - pop_data_path: file path to .csv in the data folder
    
    Output:
        - data_clean: Cleaned pandas dataframe, ready for more
                      processing and analysis.
    """
    raw_data = pd.read_csv(pop_data_path)

    # Make a copy in case I mutate the original dataframe and want it
    # back later or something
    data_clean = raw_data.copy()

    # From current pop data, this is the row that just states the 
    # overall california population data, not useful for our cases 
    # right now
    data_clean = data_clean.drop(index=data_clean.index[0])

    # Current first column values have a . before county name, 
    # i.e .Fresno and also removes the California tag for 
    # the county (as it was already downlaoded for California counties)
    data_clean['Geographic Area'] = data_clean['Geographic Area'].str.strip().str.replace(".", "", regex=False).str.replace(", California", "", regex=False) 
    
    # Set the index so I can do inner joins better for the 
    # 2010 to 2020 dataset
    data_clean = data_clean.set_index("Geographic Area")
    return data_clean 

def combine_pop_datasets(pop_path1: str, pop_path2: str, overlapAt2010: bool = True) -> pd.DataFrame:
    """
    Combine_Pop_Datasets: This function is responsible for creating two cleaned
    pandas dataframes of population data and executing an inner join in order
    to fuse the two datasets from 2000 to 2020. 

    Inputs:
        - pop_path1: Path to the first population data CSV file
        - pop_path2: Path to the second population data CSV file
        - overlapAt2010: Boolean to tell if there is an overlapping column at 2010
    
    Outputs:
        - combined_pop_df: Combined population dataframe (if bool is true)
    """
    pop1_df = clean_pop_dataframe(pop_path1)
    pop2_df = clean_pop_dataframe(pop_path2)

    if overlapAt2010:
        pop1_df = pop1_df.drop(columns=['2010'])
    
    combined_pop_df = pop1_df.join(pop2_df, how='inner')
    return combined_pop_df

def grab_specific_county_data(df: pd.Dataframe, county_name: str, start_year: str, end_year: str) -> pd.Series:
    """
    Grab_Specific_County_Data: This function reports back the population data 
    for a given county over a specified year span (from 2000 to 2020 max)
    
    Inputs:
        - df: Dataframe of the (combined, hopefully) population datasets. 
              Doesn't actually need to be, to be honest
        - county_name: Name of the county to pull data for
        - start_year: starting year for the query, as a string
        - end_year: ending year of the query, as a string
    
    Outputs:
        - queried_data: pd.Series object with the population data for 
                        that given county from the specified year range
    """

    year_cols = df.columns.astype(str)
    if start_year not in year_cols or end_year not in year_cols:
        raise ValueError(f"Start year {start_year} or End year {end_year} are out of bounds")
    
    if county_name not in df.index:
        raise ValueError(f"County Name {county_name} not in Population Data")
    
    return df.loc[county_name, start_year:end_year]

def combine_vf_wildfire_pop_data(vf_path: str, fire_path: str, popdata_path:str, county_name:str) -> pd.DataFrame:
    vf_fire_combined = combine_vf_wildfire_data(fire_path, vf_path, county_name)

if __name__ == "__main__":
    print(" --- Testing Data Loader Utilities ---")
    pop2000_2010_path = "Project/data/cali_county_pop_2000_2010.csv"
    pop2010_2020_path = 'Project/data/cali_county_pop_2010_2020.csv'
    print(" --- Clean Pop Dataframe Test ---")
    pop1 = clean_pop_dataframe(pop2000_2010_path)
    print(pop1.head())
    print(" --- END Pop Dataframe Test --- ")
    print(" --- Test Combined Pop Dataframes --- ")
    combined_df = combine_pop_datasets(pop_path1=pop2000_2010_path, pop_path2=pop2010_2020_path, overlapAt2010=True)
    print(combined_df.head())
    print(combined_df.loc["Fresno County"])
    

