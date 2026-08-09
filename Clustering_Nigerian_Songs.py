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

"""df = df[df["artist_top_genre"] != "Missing"]
top = df["artist_top_genre"].value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=top.index, y=top.values)
plt.xticks(rotation=45)
plt.title("Top genres", color="blue")"""

# plt.show()

# Concentrating on the top 3 genres, also removing the 0 popularity songs
df = df[
    (df["artist_top_genre"] == "afro dancehall")
    | (df["artist_top_genre"] == "afropop")
    | (df["artist_top_genre"] == "nigerian pop")
]
df = df[(df["popularity"] > 0)]
top = df["artist_top_genre"].value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=top.index, y=top.values)
plt.xticks(rotation=45)
plt.title("Top genres", color="blue")

corrmat = df.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)

sns.set_theme(style="ticks")

g = sns.jointplot(
    data=df,
    x="popularity",
    y="danceability",
    hue="artist_top_genre",
    kind="kde",
)

sns.FacetGrid(df, hue="artist_top_genre", height=5).map(
    plt.scatter, "popularity", "danceability"
).add_legend()

plt.show()
