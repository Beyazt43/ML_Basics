from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

# import and create a Conll extractor to use later
extractor = ConllExtractor()

# later when you need a noun phrase extractor:
"""user_input = input(" > ")
print()
print("Here are the noun phrases in your input:")
print()
user_input_blob = TextBlob(
    user_input, np_extractor=extractor
)  # note non-default extractor specified
np = user_input_blob.noun_phrases
print(np)"""

# they removed the translate method from textblob
"""blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))"""

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))

# blob.sentiment gives both polarity and subjectivity
# but you can choose to get only one of them by
# blob.sentiment.polarity or blob.sentiment.subjectivity
