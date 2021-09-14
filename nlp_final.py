import json
import nltk.stem
import nltk
from sklearn import svm
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import classification_report
# uncomment line below if wordnet not found, only need to run with it once
nltk.download('wordnet')

features = {}
def main():
    trainData = []
    testData = []
    # percentile of test data
    testratio = .9
    sent_path = "training.1600000.processed.noemoticon.csv"
    ama_path = "reviews.json"
    print("Loading data from %s." % ama_path)
    print("Loading data from %s." % sent_path)
    data = parse_json_reviews(ama_path)
    sent_data = parse_csv_to_list("training.1600000.processed.noemoticon.csv")
    shuffle(sent_data)
    # trim input if needed for test
    #sent_data = sent_data[:100000]
    if not data[0].get("reviewerID"):
        print("Error: data failed to be extracted to a usable format.")
        exit(1)
    print("Data load complete.")
    print("Splitting data into test and training portions.")
    count = 0
    '''
    # used to validate alternate json input data
    for data_dict in data:
        train_dict = word_to_vector(data_dict.get("summary")+data_dict.get("reviewText"))
        count += 1
        # 90% of entries used for training, the other 10% are used for test
        if count / len(data) > testratio:
            testData.append((train_dict, impression(data_dict.get("overall"))))
        else:
            trainData.append((train_dict, impression(data_dict.get("overall"))))
    print("Split complete.")
    '''
    for data_list in sent_data:
        train_dict = word_to_vector(data_list[5])
        count += 1
        # 90% of entries used for training, the other 10% are used for test
        if count / len(sent_data) > testratio:
            testData.append((train_dict, impression(int(data_list[0]))))
        else:
            trainData.append((train_dict, impression(int(data_list[0]))))
    print("Split complete.")

    print("Sample of test data: ")
    print(testData[:3])

    print("Test size: ", len(testData))
    print("Train size: ", len(trainData))

    # choose classifier (nb,dt,linearsvc) and
    cv_results = crossValidate(trainData, 10, "linearsvc")
    print("Classifier accuracy when using review rating.")
    # Precision – Accuracy of positive predictions.
    # Recall (aka sensitivity or true positive rate): Fraction of positives That were correctly identified.

    print("", cv_results[0])
    print("Processed %d reviews: %d training, %d test" % (len(data), len(trainData), len(testData)))
    print("Feature set size: %d" % len(features))

    # plot it
    #fd = nltk.FreqDist(features)
    #fd.plot()
    #print(features['good'])
    #print(max(features, key=features.get()))
    #plt.bar(list(features.keys()), features.values(), color='b')
    #plt.show()

def impression(overall):
    if overall > 3:
        return "Positive"
    else:
        return "Negative"

def parse_csv_to_list(path):
    """
    The data is a CSV with emoticons removed. Data file format has 6 fields:
    0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    1 - the id of the tweet (2087)
    2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    3 - the query (lyx). If there is no query, then this value is NO_QUERY.
    4 - the user that tweeted (robotickilldozr)
    5 - the text of the tweet (Lyx is cool)
    """
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def parse_json_reviews(path):
    """
    example:
    {"reviewerID": "A11N155CW1UV02", "asin": "B000H00VBQ", "reviewerName": "AdrianaM", "helpful": [0, 0],
    #"reviewText": "asdf.",
    #"overall": 2.0, "summary": "A little bit boring for me", "unixReviewTime": 1399075200, "reviewTime": "05 3, 2014"}
    """
    data = []
    for line in open(path, 'r'):
        data.append(json.loads(line))
    return data

def crossValidate(dataset, folds, cl):
    shuffle(dataset)
    results = []
    foldSize = int(len(dataset) / folds)  # it wants an int instead of float
    for i in range(0, len(dataset), foldSize):
        crossTestData = dataset[i:i + foldSize]
        crossTrainData = dataset[:i] + dataset[i + foldSize:]
        # Naive Bayes classifier
        if cl == "nb":
            print("Train Naive Bayes classifier fold %d..." % i)
            classifier = nltk.classify.NaiveBayesClassifier.train(crossTrainData)
        # linear SVC
        elif cl == "linearsvc":
            print("Train linearsvc classifier fold %d..." % i)
            pipeline = Pipeline([('svc', svm.LinearSVC(C=0.01, class_weight=None, verbose=0, dual=True, fit_intercept=True,
                                                   intercept_scaling=1, loss='squared_hinge', max_iter = 10000,
                                                   multi_class='ovr', penalty='l2', random_state=0, tol=0.0001))])
            # build classifier
            classifier = SklearnClassifier(pipeline).train(crossTrainData)
        # Decision Tree classifier
        elif cl == "dt":
            # times out on larger sets, don't use
            print("Train Decision Tree classifier fold %d..." % i)
            classifier = nltk.classify.DecisionTreeClassifier.train(crossTrainData, entropy_cutoff = 0)
        else:
            print("failed to define classifier type.")
            raise TypeError
        y_true = [x[1] for x in crossTestData]
        # get predictions on test set
        y_pred = classifier.classify_many(map(lambda t: t[0], crossTestData))
        results.append(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
    return results

# convert words to vectors and format text
def word_to_vector(text):
    vector_dict = {}
    rm = RegexpTokenizer(r'[a-zA-Z]\w+\'?\w*')
    # remove meta chars
    rm.tokenize(text)
    english_stemmer = nltk.stem.SnowballStemmer('english')

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda text: (english_stemmer.stem(w) for w in analyzer(text))

    stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
    stem_analyze = stem_vectorizer.build_analyzer()
    tokens = stem_analyze(text)

    for token in tokens:
        # ignore these common points
        ignore = ['8217', '!).', ':', 'http', '.', ',', '?', '...', "'s", "n't", 'RT', ';', '&', ')', '(', '``', 'u', '(', "''", '|', '!']
        token = WordNetLemmatizer().lemmatize(token)
        # add each word to a global dict, and a local one for classifier consumption
        if token not in ignore:
            if token[0:2] != '//':
                if isinstance(token, str):
                    if token in vector_dict:
                        vector_dict[token] += 1
                        features[token] += 1
                    else:
                        vector_dict[token] = 1
                        features[token] = 1
    return vector_dict

if __name__ == "__main__":
    main()