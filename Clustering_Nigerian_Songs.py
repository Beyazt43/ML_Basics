import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

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
"""plt.figure(figsize=(10, 7))
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
).add_legend()"""

# plt.show()

plt.figure(figsize=(20, 20), dpi=200)

plt.subplot(4, 3, 1)
sns.boxplot(x="popularity", data=df)

plt.subplot(4, 3, 2)
sns.boxplot(x="acousticness", data=df)

plt.subplot(4, 3, 3)
sns.boxplot(x="energy", data=df)

plt.subplot(4, 3, 4)
sns.boxplot(x="instrumentalness", data=df)

plt.subplot(4, 3, 5)
sns.boxplot(x="liveness", data=df)

plt.subplot(4, 3, 6)
sns.boxplot(x="loudness", data=df)

plt.subplot(4, 3, 7)
sns.boxplot(x="speechiness", data=df)

plt.subplot(4, 3, 8)
sns.boxplot(x="tempo", data=df)

plt.subplot(4, 3, 9)
sns.boxplot(x="time_signature", data=df)

plt.subplot(4, 3, 10)
sns.boxplot(x="danceability", data=df)

plt.subplot(4, 3, 11)
sns.boxplot(x="length", data=df)

plt.subplot(4, 3, 12)
sns.boxplot(x="release_date", data=df)

plt.show()

le = LabelEncoder()

X = df.loc[
    :,
    (
        "artist_top_genre",
        "popularity",
        "danceability",
        "acousticness",
        "loudness",
        "energy",
    ),
]

y = df["artist_top_genre"]

X["artist_top_genre"] = le.fit_transform(X["artist_top_genre"])

y = le.transform(y)

nclusters = 3
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X)

# Predict the cluster for each data point

y_cluster_kmeans = km.predict(X)
y_cluster_kmeans
