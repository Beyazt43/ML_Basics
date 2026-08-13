import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("stopwords")
nltk.download("vader_lexicon")

vader_sentiment = SentimentIntensityAnalyzer()


# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]


df = pd.read_csv("data/Hotel_Reviews_Filtered.csv")
# We want to find the most useful tags to keep
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex=False)
# removing this to take advantage of the 'already a phrase' fact of the dataset
# Now split the strings into a list
tag_list_df = df.Tags.str.split(",", expand=True)
# Remove leading and trailing spaces
df["Tag_1"] = tag_list_df[0].str.strip()
df["Tag_2"] = tag_list_df[1].str.strip()
df["Tag_3"] = tag_list_df[2].str.strip()
df["Tag_4"] = tag_list_df[3].str.strip()
df["Tag_5"] = tag_list_df[4].str.strip()
df["Tag_6"] = tag_list_df[5].str.strip()
# Merge the 6 columns into one with melt
df_tags = df.melt(value_vars=["Tag_1", "Tag_2", "Tag_3", "Tag_4", "Tag_5", "Tag_6"])
# Get the value counts
tag_vc = df_tags.value.value_counts()
# print(tag_vc)
"""print("The shape of the tags with no filtering:", str(df_tags.shape))"""
# Drop rooms, suites, and length of stay, mobile device and anything with less count than a 1000
df_tags = df_tags[
    ~df_tags.value.str.contains(
        "Standard|room|Stayed|device|Beds|Suite|Studio|King|Superior|Double",
        na=False,
        case=False,
    )
]
tag_vc = df_tags.value.value_counts().reset_index(name="count").query("count > 1000")
# Print the top 10 (there should only be 9 and we'll use these in the filtering section)
"""print(tag_vc[:10])"""


start = time.time()
cache = set(stopwords.words("english"))


def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text


# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"data/Hotel_Reviews_NLP.csv", index=False)


# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")

df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])

# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(
    [
        "Hotel_Name",
        "Hotel_Address",
        "Total_Number_of_Reviews",
        "Average_Score",
        "Reviewer_Score",
        "Negative_Sentiment",
        "Positive_Sentiment",
        "Reviewer_Nationality",
        "Leisure_trip",
        "Couple",
        "Solo_traveler",
        "Business_trip",
        "Group",
        "Family_with_young_children",
        "Family_with_older_children",
        "With_a_pet",
        "Negative_Review",
        "Positive_Review",
    ],
    axis=1,
)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"data/Hotel_Reviews_NLP.csv", index=False)
