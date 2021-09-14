# NLP_sentiment_analysis

How do you determine fake reviews?

- One method is by analyzing sentiment

- Separate negative and positive reviews

- Weight words in those reviews with their respective
rating types

- Then train a classifier based on this data!

- Use classifier to find reviews with ratings that don’t
match their sentiment

## Approach
- Obtain a repository of Amazon reviews and metadata
- Import data in json format
- Extract text from positive and negative reviews
- Sanitize text (remove metacharacters and links)
- Use nltk lib (natural language tool kit) to process
English words
- Use Sklearn’s StemmedCountVectorizer to stem and
count words (“fish” and “fishes” becomes {“fish” :2})
- Use nltk to lemmatize (normalizes variants of words,
“fish” and “fishing” become {“fish” : 2})
- Compile these results into a vector of features
(dictionary of words with their counts in Pythonese)
- Repeat for each review, 90% of which used to train
the classifier

## Example

This review…
{"reviewerID": "A11N155CW1UV02", "asin": "B000H00VBQ",
"reviewerName":
"AdrianaM", "helpful": [0, 0], "reviewText": “THE
TEXT HERE!!!", "overall": 2.0,
"summary": "A little bit boring for me",
"unixReviewTime": 1399075200,
"reviewTime": "05 3, 2014"}

Becomes…
Negative_dict = { “Text” : 1, “Here” : 1}


## Approach (continued)

- Train the classifier over five runs (aka folds) of the
training data (divided into equal portions) using
“cross validation”
- Use sklearn LinearSVC (support vector classifier) to
train a classifier
- Features (words) are used to train classifier on
samples (reviews) based on token vectors (frequency
of each word)
- SklearnClassifier, using LinearSVC method, is trained on this data
- After final fold, classifier precision (accuracy of
positive predictions) and recall (fraction of positives
that were correctly identified) are determined
- Samples that are do not match model are more likely to be fake

## Results
Using 37126 reviews: 90% training, 10% test
86% precision
Feature size: 50821
