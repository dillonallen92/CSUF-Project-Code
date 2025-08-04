import sys
from pathlib import Path 
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loader import combine_vf_wildfire_data, combine_vf_fire_pop_data

def load_combined_data(county : str, use_pop : bool = True, bInterp: bool = False, 
                       bConvRate:bool = False) -> pd.DataFrame:
  if use_pop:
    return combine_vf_fire_pop_data(
      pop1_path ="Project/data/cali_county_pop_2000_2010.csv",
      pop2_path = "Project/data/cali_county_pop_2010_2020.csv",
      vf_cases_path = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv",
      wildfire_path = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv",
      county = county, 
      start_year= "2006", 
      end_year = "2015",
      bInterp = bInterp,
      bConvRate= bConvRate
    )
  else:
    return combine_vf_wildfire_data(
      vf_cases_path = "Project/data/coccidioidomycosis_m2000_2015_v0.1.csv",
      fire_path = "Project/data/CAL_FIRE_Wildland_PublicReport_2000to2018.csv",
      county_name = county
    )

if __name__ == "__main__":
  print(" --- With Pop --- ")
  df1 = load_combined_data("Fresno", True)
  print(df1)
  print(" --- Without Pop --- ")
  df2 = load_combined_data("Fresno", False)
  print(df2)
