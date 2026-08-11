# Load the hotel reviews from CSV
import time

import pandas as pd

# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()

# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv("data/Hotel_Reviews.csv")
end = time.time()

print("Loading took " + str(round(end - start, 2)) + " seconds")
# print(df.shape)
# number of unique nationalities in this dataset (227)
nationality_freq = df["Reviewer_Nationality"].value_counts()
"""print("size = " + str(nationality_freq.size))
print(nationality_freq)"""
"""
# What is the top 10 most common nationalities and their frequencies?
print("The next 10 highest frequency reviewer nationalities are:")
print(nationality_freq[0:10].to_string())"""

for nat in nationality_freq[:10].index:
    # First, extract all the rows that match the criteria into a new dataframe
    nat_df = df[df["Reviewer_Nationality"] == nat]
    # Now get the hotel freq
    freq = nat_df["Hotel_Name"].value_counts()
    print(
        "The most reviewed hotel for "
        + str(nat).strip()
        + " was "
        + str(freq.index[0])
        + " with "
        + str(freq.iloc[0])
        + " reviews."
    )
# freq.iloc was necessary to get the first row, not the one with the label 0
