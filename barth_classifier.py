# Based very heavily on
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
# from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import numpy
import os
from pandas import DataFrame
from spacy.en import English
import string

# Create the parser with spaCy
parser = English()

# Use default nltk stoplist for now
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

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
    def transform(self, X, **transform_params):
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
    text = text[0].strip().replace('\n', ' ').replace('\r', ' ')

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
vectorizer = CountVectorizer(ngram_range=(1, 1))
# in above vectorizer, need to get custom tokenizer with spacy working
classifier = LinearSVC()
# create a pipline that cleans, tokenizes and vectorizes, and classifies
pipeline = Pipeline([
    ('cleanText', CleanTextTransformer()),
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])


# read in data
# i want to have two lists, one of text, the other of correspond labels
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            # cycle back through for subdirs
            read_files(os.path.join(root, path))
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                text = []
                with open(file_path, encoding='latin-1') as f:
                    for line in f:
                        text.append(line)
                yield file_path, text


# use pandas and build a dataframe, because pandas is nice
def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        # use the file names for the indices
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

BEFORE = 'before'
AFTER = 'after'

SOURCES = [
    ('paragraphs/pre-election', BEFORE),
    ('paragraphs/post-election', AFTER)
]

data = DataFrame({'text': [], 'class': []})
for (path, classification) in SOURCES:
    print(path, classification)
    data = data.append(build_data_frame(path, classification))

data.to_csv('data.csv')

# shuffle the dataset to help with validating prediction accuracy
data = data.reindex(numpy.random.permutation(data.index))

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=AFTER)
    scores.append(score)

print('total texts classified: ', len(data))
print('Score: ', sum(scores)/len(scores))
print('Confusion matrix: ')
print(confusion)
