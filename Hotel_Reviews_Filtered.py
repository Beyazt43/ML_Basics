import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("stopwords")
nltk.download("vader_lexicon")
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
