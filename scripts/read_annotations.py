import pandas as pd
from io import StringIO


# Read the CSV data into a DataFrame
df = pd.read_csv("data/annotations.csv")

# Initialize a dictionary to store the results
result = {}

# Iterate over the columns, starting from the second column (index 1)
for column in df.columns[1:]:
    # Iterate over each row in the current column
    for index, value in df[column].items():
        if pd.notna(value):  # Check if the value is not NaN
            # Split the value by comma and strip any whitespace
            attributes = [attr.strip() for attr in value.split(",")]
            for attribute in attributes:
                # Add the Natbib value to the corresponding attribute in the result
                if column not in result:
                    result[column] = {}
                if attribute not in result[column]:
                    result[column][attribute] = []
                result[column][attribute].append(df.loc[index, "Natbib"])

# Print the results
for category, attributes in result.items():
    # print(f"Category: {category}")
    for attribute, natbibs in attributes.items():
        print(f"{category}: {attribute}\n\t\\citep{{{', '.join(natbibs)}}}")
