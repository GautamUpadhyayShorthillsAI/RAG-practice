import pandas as pd

# Load the CSV file and clean column names
df = pd.read_csv("all_test_cases.csv")
df.columns = df.columns.str.strip().str.lower()  # Normalize column names

# Print available columns
print("Columns in CSV:", df.columns.tolist())  

# Check if required columns exist
required_columns = {"expected_answer", "col1", "col2", "col3", "col4"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    print(f"Error: Missing columns in CSV - {missing_columns}")
else:
    # Merge columns into 'expected_answer'
    df["expected_answer"] = df["expected_answer"].astype(str) + " " + \
                            df["col1"].astype(str) + " " + \
                            df["col2"].astype(str) + " " + \
                            df["col3"].astype(str) + " " + \
                            df["col4"].astype(str)

    # Drop unnecessary columns
    df.drop(columns=["col1", "col2", "col3", "col4"], inplace=True)

    # Save the updated CSV
    df.to_csv("merged_file.csv", index=False)
    print("Columns merged successfully!")
