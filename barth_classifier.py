# Based very heavily on
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from spacy.en import English
import string

# Create the parser with spaCy
parser = English()

# Use default nltk stoplist for now
STOPLIST = set(stopwords.s('english') + list(ENGLISH_STOP_WORDS))

# Create a group of symbols to remove
SYMBOLS = " ".join(string.punctuation).split(" ") + \
    ["-----", "---", "...", "“", "”", "'ve"]


# Create a transformer to clean the text with spaCy
# This transformer inherits from the base in sklearn
class CleanTextTransformer(TransformerMixin):
    """
    Cleans raw text
    """

    # Here, X is the list of texts?
    def tranform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    # Since we're not actually fitting data since this is an intermediate step
    # in the pipeline, we just return self
    def fit(self, X, y=None, **fit_params):
        return self

    # not sure what get_params does - will look up
    def get_params(self, deep=True):
        return {}


# Custom function to clean the text
def cleanText(text):
    # get rid of newlines/carriage returns
    text = text.strip().replace('\n', ' ').replace('\r', ' ')

    # given text scraped from the web, remove HTML symbols
    text = (
        text.replace('&amp;', 'and')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
    )

    # lowercase
    text = text.lower()

    return text


# Tokenize the text with spaCy
# could consider stemming or lemmatization for classifier
def tokenizeText(text):
    # get tokens
    tokens = parser(text)

    # stoplist the tokens
    tokens = [token for token in tokens if token not in STOPLIST]

    # stoplist the symbols
    tokens = [token for token in tokens if token not in SYMBOLS]

    return tokens


def printNMostInformative(vectorizer, classifier, N):
    """
    Prints features with the highest coefficient values per class
    """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N+1):-1]
    print("Class 1 best: ")
    for feature in topClass1:
        print(feature)
    for feature in topClass2:
        print(feature)


# define the vectorizer and classifier
# with the CountVectorizer, use the custom tokenizer function
# go ahead and specify n-gram range to change later if desired
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1, 1))
classifier = LinearSVC()
# create a pipline that cleans, tokenizes and vectorizes, and classifies
pipe = Pipeline([
    ('cleanText', CleanTextTransformer()),
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

# read in data
# i want to have two lists, one of text, the other of correspond labels
