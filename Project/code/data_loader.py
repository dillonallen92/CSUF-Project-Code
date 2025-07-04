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