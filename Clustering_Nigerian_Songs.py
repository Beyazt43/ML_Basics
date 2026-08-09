import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/nigerian-songs.csv")
# df.info()
# print(df.isnull().sum())
# DataFrame.describe() method is used to generate descriptive statistics that summarize the central tendency, dispersion and the shape of a dataset, excluding NaN values
# print(df.describe())

"""top = df["artist_top_genre"].value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=top[:5].index, y=top[:5].values)
plt.xticks(rotation=45)
plt.title("Top genres", color="blue")"""

# getting rid of the Missing genre

df = df[df["artist_top_genre"] != "Missing"]
top = df["artist_top_genre"].value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=top.index, y=top.values)
plt.xticks(rotation=45)
plt.title("Top genres", color="blue")

plt.show()
