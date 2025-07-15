import pandas as pd 

# Fire Data
# Fire Data is of the from 2006 - 2018. 
# Months are missing, so need to replace with zeros for those values
fire_data_2018_path = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv"
fire_2018_data = pd.read_csv(fire_data_2018_path, low_memory=False)

# specify to Tulare County
fd_2018_tul = fire_2018_data[fire_2018_data['County'] == 'Tulare']

# VF Data
# Goes from 2000 - 2015, with each month as a column in the dataframe
# Need to transpose this dataframe so we can do a monthly count
vf_data_path = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv"
vf_data      = pd.read_csv(vf_data_path, low_memory=False)
vf_tul       = vf_data[vf_data['County'] == 'Tulare']

########################################
# Turn the Fire Data into Month Counts #
########################################
fd_2018_tul["Incident Date"] = pd.to_datetime(fd_2018_tul["Incident Date"], errors="coerce")

# Filter from Jan 2006 to Dec 2018
mask = (fd_2018_tul["Incident Date"] >= "2006-01-01") & (fd_2018_tul["Incident Date"] <= "2018-12-31")
filtered_df = fd_2018_tul.loc[mask]

# Set datetime index
filtered_df = filtered_df.set_index("Incident Date")

# Resample to get count per month
monthly_counts = filtered_df.resample("M").size()

# Fill in missing months with 0
all_months = pd.date_range(start="2006-01-01", end="2018-12-31", freq="M")
monthly_counts = monthly_counts.reindex(all_months, fill_value=0)

# Clean up to final DataFrame
monthly_counts_df = monthly_counts.reset_index()
monthly_counts_df.columns = ["Month", "Fire Incident Count"]

monthly_counts_df["Month"] = monthly_counts_df["Month"].dt.to_period("M").astype(str)

fd_tul_monthly = monthly_counts_df
print(fd_tul_monthly)

########################################
#    Convert the Valley Fever Data     #
########################################
# Drop metadata columns
data = vf_tul.drop(columns=["State", "County", "FIPS"])

# Transpose and reset index
monthly_counts = data.T.reset_index()

# Rename columns
monthly_counts.columns = ["Month", "VF Case Count"]

# Optional: convert 'Month' to datetime
monthly_counts["Month"] = pd.to_datetime(monthly_counts["Month"], format="%Y/%m")
monthly_counts["Month"] = monthly_counts["Month"].dt.to_period("M").astype(str)

vf_data_pd = monthly_counts

###########################
#   Fuse the Dataframes   #
###########################
# combine
# Ensure 'Month' columns are strings in YYYY-MM format
monthly_counts_df["Month"] = monthly_counts_df["Month"].astype(str)
monthly_counts["Month"] = monthly_counts["Month"].astype(str)

# Filter both to the range 2006-01 to 2015-12
fire_trim = monthly_counts_df[
    (monthly_counts_df["Month"] >= "2006-01") & (monthly_counts_df["Month"] <= "2015-12")
]
vf_trim = monthly_counts[
    (monthly_counts["Month"] >= "2006-01") & (monthly_counts["Month"] <= "2015-12")
]

# Merge on Month
combined_df = pd.merge(fire_trim, vf_trim, on="Month", how="inner")

# View result
vf_fire_dataset_combined = combined_df

# Note to self - Need to make a more efficient pipeline but 
# need more data. 
