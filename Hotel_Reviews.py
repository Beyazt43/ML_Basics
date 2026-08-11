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


# First create a new dataframe based on the old one, removing the uneeded columns
hotel_freq_df = df.drop(
    [
        "Hotel_Address",
        "Additional_Number_of_Scoring",
        "Review_Date",
        "Average_Score",
        "Reviewer_Nationality",
        "Negative_Review",
        "Review_Total_Negative_Word_Counts",
        "Positive_Review",
        "Review_Total_Positive_Word_Counts",
        "Total_Number_of_Reviews_Reviewer_Has_Given",
        "Reviewer_Score",
        "Tags",
        "days_since_review",
        "lat",
        "lng",
    ],
    axis=1,
)

# Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
"""hotel_freq_df["Total_Reviews_Found"] = hotel_freq_df.groupby("Hotel_Name").transform(
    "count"
)

# Get rid of all the duplicated rows
hotel_freq_df = hotel_freq_df.drop_duplicates(subset=["Hotel_Name"])
hotel_freq_df = hotel_freq_df.sort_values(by="Total_Reviews_Found", ascending=False)"""

hotel_freq_df = (
    df.groupby("Hotel_Name")
    .size()
    .reset_index(name="Total_Reviews_Found")
    .sort_values("Total_Reviews_Found", ascending=False)
)
print(hotel_freq_df)


# define a function that takes a row and performs some calculation with it
def get_difference_review_avg(row):
    return row["Average_Score"] - row["Calc_Average_Score"]


# 'mean' is mathematical word for 'average'
df["Calc_Average_Score"] = round(
    df.groupby("Hotel_Name").Reviewer_Score.transform("mean"), 1
)

# Add a new column with the difference between the two average scores
df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis=1)

# Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
review_scores_df = df.drop_duplicates(subset=["Hotel_Name"])

# Sort the dataframe to find the lowest and highest average score difference
review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])

print(
    review_scores_df[
        [
            "Average_Score_Difference",
            "Average_Score",
            "Calc_Average_Score",
            "Hotel_Name",
        ]
    ]
)


# without lambdas (using a mixture of notations to show you can use both)
start = time.time()
no_negative_reviews = sum(df.Negative_Review == "No Negative")
print("Number of No Negative reviews: " + str(no_negative_reviews))

no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
print("Number of No Positive reviews: " + str(no_positive_reviews))

both_no_reviews = sum(
    (df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive")
)
print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))

end = time.time()
print("Sum took " + str(round(end - start, 2)) + " seconds")
