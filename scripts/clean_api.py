import pandas as pd
from pathlib import Path

# Deletes unwanted manipulator values

# Change path for input and output files
input_data = Path("C:\\Users\\Tracy\\Documents\\2026 RCL UBC\\xi_api_data 05142026\\api\\KinCsvLog_14-05-2026_20-09-43.699.csv")
cleaned_data = Path("C:\\Users\\Tracy\\Projects\\Xi_EE_Pose\\data\\cleaned_data.csv")

#csv to dataframe, and skip initial space when going acorss columns
df = pd.read_csv(input_data, skipinitialspace=True) 

df["manip"] = pd.to_numeric(df["manip"], errors="coerce")

remove_manip_values = [3, 4, 7, 8]

cleaned_df = df[~df["manip"].isin(remove_manip_values)]

cleaned_df.to_csv(cleaned_data, index=False)

print("Data cleaning complete. removed manip values for: " + str(remove_manip_values))