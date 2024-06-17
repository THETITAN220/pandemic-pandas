import pandas as pd

# Import CSV into a DataFrame
df = pd.read_csv("C:\Users\itsra\OneDrive\Documents\Datasets\archive\ebola_2014_2016_clean.csv")

# Get the summary statistics
statistics = df.describe()

# Display the statistics
print(statistics)