import pandas as pd

# Load the in-house dataset
df1 = pd.read_csv('preprocessed_InHouseData.csv')

# Load the baseline dataset
df2 = pd.read_csv('2022-03-31-20aug-1cm-720-1800-20baseline.csv')

# Standardize the 'plastic' column name to 'Plastic'
df2.rename(columns={'plastic': 'Plastic'}, inplace=True)

# Replace full plastic names with abbreviations in the baseline dataset
df2 = df2.replace(
    ['acrylonitrile butadiene styrene', 'poly (methyl methacrylate)', 'polycarbonate', 'polyester', 'polyethylene', 'polyethylene terephthalate', 'polyoxymethylene', 'polypropylene', 'polystyrene', 'polytetrafluoroethylene', 'polyurethane', 'polyvinyl chloride'],
    ['ABS', 'PMMA', 'PC', 'PES', 'PE', 'PET', 'POM', 'PP', 'PS', 'PTFE', 'PU', 'PVC']
)

# Remove the 'sampleid' and 'supplier' columns from df2 (baseline dataset)
df2 = df2.drop(columns=['sampleid', 'supplier'])

# Check column consistency between the two datasets
if set(df1.columns) == set(df2.columns):
    print("Columns match. Proceeding to combine datasets.")
else:
    print("Columns do not match. Check the datasets.")

print("Last few rows of df1:")
print(df1.tail())

print("First few rows of df2:")
print(df2.head())


# Combine the two datasets by rows
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Save the combined dataset
combined_df.to_csv('combined_data_2.csv', index=False)

print("Datasets combined successfully.")

# Try reading the file again
try:
    df_test = pd.read_csv('combined_data_2.csv')
    print("File loaded successfully. Here are the first few rows:")
    print(df_test.head())
except Exception as e:
    print(f"Error loading file: {e}")

print("End of script.")

