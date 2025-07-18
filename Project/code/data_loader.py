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

def grab_specific_county_data(df: pd.DataFrame, county_name: str, start_year: str, end_year: str) -> pd.Series:
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

def pop_copy_monthly(df: pd.DataFrame, pop_series: pd.Series, col_name:str) -> pd.DataFrame:
    """
    Pop_Copy_Monthly: This function takes in a dataframe and copies the population
    data from the pop_series (a yearly value) for every month value in the dataframe.

    Inputs:
        - df (pd.DataFrame): Pandas DataFrame of data (ex: wildfire and virus data).
                             This data is typically reported (and indexed) on a monthly
                             scale.
        - pop_series (pd.Series): Population data in a pd.Series structure. 
                                  Each index in the series represents a single year
        - col_name (string): string value of the column name we want to copy
    
    Outputs:
        - combined_pop_monthly_data (pd.DataFrame): Pandas dataframe containing the 
                                                    population data copied onto a monthly basis.
    """    
    df[col_name] = df.index.year.map(pop_series)

    # Step 4: Sanity check for missing population values
    if df[col_name].isnull().any():
        missing_years = df[df[col_name].isnull()].index.year.unique()
        raise ValueError(f"Missing {col_name} data for years: {missing_years.tolist()}")

    combined_pop_monthly_data = df.copy()
    return combined_pop_monthly_data 

def combine_vf_fire_pop_data(pop1_path: str, pop2_path: str, vf_cases_path: str, wildfire_path: str,
                             county: str, start_year: str = "2006", end_year: str = "2015", 
                             bInterp: bool = False) -> pd.DataFrame:
    """
    Combine_VF_Fire_Pop_Data: Combines VF case data, wildfire data, and annual population data for a given county
    into a single monthly DataFrame with aligned population values.

    Inputs:
        - pop1_path (str): Path to 2000–2010 population CSV
        - pop2_path (str): Path to 2010–2020 population CSV
        - vf_cases_path (str): Path to VF case data CSV
        - wildfire_path (str): Path to wildfire incident data CSV
        - county (str): County name, e.g. "Fresno"
        - start_year (str): Starting year as string, default "2006"
        - end_year (str): Ending year as string, default "2015"
        - bInterp (bool): Boolean flag for population interpolation capability

    Output:
        vf_fire_pop_df (pd.DataFrame): Combined monthly VF, wildfire, and population dataset into a pandas DataFrame
    """

    # Format full county name for population match
    county_name = f"{county} County"

    # Step 1: Combine VF + wildfire data
    vf_fire_df = combine_vf_wildfire_data(wildfire_path, vf_cases_path, county)
    vf_fire_df = vf_fire_df.copy()
    vf_fire_df = vf_fire_df.set_index('Month')
    vf_fire_df.index = pd.to_datetime(vf_fire_df.index)

    # Step 2: Combine and extract population data
    pop_combined = combine_pop_datasets(pop1_path, pop2_path, True)
    pop_series = grab_specific_county_data(pop_combined, county_name, start_year, end_year)
    pop_series = pop_series.str.replace(",", "").astype(int)
    pop_series.index = pop_series.index.astype(int)

    # Step 3: Map population to monthly data
    if not bInterp:
        vf_fire_pop_df = pop_copy_monthly(vf_fire_df, pop_series, "Population")
    
    return vf_fire_pop_df


if __name__ == "__main__":
    """
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
    print(combined_df.loc["Fresno County"]) """

    # Import Data
    pop2000_2010_path = "Project/data/cali_county_pop_2000_2010.csv"
    pop2010_2020_path = 'Project/data/cali_county_pop_2010_2020.csv'
    vf_cases_path     = 'Project/data/coccidioidomycosis_m2000_2015_v0.1.csv'
    wildfire_path     = 'Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv'
    county_name       = "Fresno"
    start_year        = "2006"
    end_year          = "2015"

    print(" --- Testing Total Combination Dataset")
    comb_dataset = combine_vf_fire_pop_data(pop2000_2010_path, pop2010_2020_path,
                                            vf_cases_path, wildfire_path, county_name,
                                              start_year, end_year)
    print(comb_dataset.head())
