import pandas as pd

# Load the CSV file
df = pd.read_csv("your_file.csv")

# Merge two columns (e.g., 'Column1' and 'Column2') into a new column 'MergedColumn'
df["MergedColumn"] = df["Column2"].astype(str) + " " + df["Column3"].astype(str) + " " + df["Column4"].astype(str) + " " + df["Column5"].astype(str) + " " + df["Column6"].astype(str)  

# Save the updated CSV
df.to_csv("merged_file.csv", index=False)

print("Columns merged successfully!")
