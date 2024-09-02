import pandas as pd

# Load the data
data = pd.read_csv('compiled_preprocessed_data.csv')

# Replace old plastic names with new ones
data_replaced = data.replace(
    ['PP2', 'polyester', 'PS2', 'LDPEdoorzichtig', 'PE2', 'PP3'],
    ['PP', 'PES', 'PS', 'LDPE', 'PE', 'PP']
)

# Drop unnecessary columns
data_dropped_columns = data_replaced.drop(
    ["Filename", "Size", "Laser", "Power", "Grating", "Acq_Time_Scan_Nmb"],
    axis=1
)

# Remove rows where "Plastic" column contains specific names
rows_to_remove = ["particle1", "particle2", "particle3", "particle4", "particle5", "particle6", "onbstaal1", "onbstaal2", "onbstaal3", "onbstaal4", "titaniumdioxide"]
data_filtered = data_dropped_columns[~data_dropped_columns["Plastic"].isin(rows_to_remove)]

# Ensure the "Plastic" column is the first column
cols = ["Plastic"] + [col for col in data_filtered.columns if col != "Plastic"]
data_final = data_filtered[cols]

# Save or inspect the final DataFrame
print("First few rows of the cleaned DataFrame:")
print(data_final.head())

print("\nShape of the cleaned DataFrame:", data_final.shape)

print("\nColumn names in the cleaned DataFrame:")
print(data_final.columns)

print("\nSummary of the cleaned DataFrame:")
print(data_final.info())

print("\nSummary statistics of the features in the cleaned DataFrame:")
print(data_final.describe())

print("\nMissing values in each column of the cleaned DataFrame:")
print(data_final.isnull().sum())

# Optionally, save the cleaned DataFrame to a new CSV file
data_final.to_csv('preprocessed_InHouseData.csv', index=False)
