from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

# import and create a Conll extractor to use later
extractor = ConllExtractor()

# later when you need a noun phrase extractor:
user_input = input(" > ")
print()
print("Here are the noun phrases in your input:")
print()
user_input_blob = TextBlob(
    user_input, np_extractor=extractor
)  # note non-default extractor specified
np = user_input_blob.noun_phrases
print(np)
