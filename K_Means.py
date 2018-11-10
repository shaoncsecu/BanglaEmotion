'''
Copyright (c) 2018 MD ATAUR RAHMAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

import matplotlib.pyplot as plt
from sklearn.metrics import *
import warnings
import time
import numpy

# Filters all warnings - specially to ignore warnings due to 0.0 in precision and recall and f-measures
warnings.filterwarnings("ignore")

# Go to the last two lines of this program to have an idea from start (bottom-up functional approach)

def read_corpus(corpus_file):
    documents = []
    labels_topic = []
    labels_sentiment = []

    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()                   # tokenize the lines

            documents.append(tokens[3:])                    # append the text - starts from 4th tokens

            # 2-class problem: positive vs negative
            labels_sentiment.append( tokens[1] )                  # tokens[1] is sentiment type (either pos/neg)

            # 6-class problem: books, camera, dvd, health, music, software
            labels_topic.append( tokens[0] )                  # tokens[0] is one of 6 topic types

    return documents, labels_topic, labels_sentiment


# separates two labels from original dataset
def separate_two_labels(documents, labels_topic, labels_sentiment, sample_1, sample_2):
    docs_two = []
    labels_twoTopic = []
    labels_twoSenti = []

    for txt, lbl, senti in zip(documents, labels_topic, labels_sentiment):

        if lbl == sample_1 or lbl == sample_2:
            docs_two.append(txt)
            labels_twoTopic.append(lbl)
            labels_twoSenti.append(senti)

    # for txt, lbl, senti in zip(docs_two[0:10], labels_twoTopic, labels_twoSenti):
    #     print(lbl)
    #     print(senti)
    #     print(txt)

    return docs_two, labels_twoTopic, labels_twoSenti

# a dummy function that just returns its input
def identity(x):
    return x


# Apply Stemmer on Documents (I can't figure out how to pass it in Pipeline so done it separately)
def stem_documents(documents):
    # Apply Porter's stemming (pre-processor)
    stemmed_documents = []
    for doc in documents:
        stemmed_documents.append(apply_stemmer(doc))

    return stemmed_documents


# we are using NLTK stemmer to stem multiple words into root
def apply_stemmer(doc):
    stemmer = PorterStemmer()

    roots = [stemmer.stem(plural) for plural in doc]

    return roots


# NLTK POS Tagger
def tokenize_pos(tokens):
    # Using only NN (noun variants).
    # JJ (adjective variants) and RB (adverb variants) are also been experimented but didn't improve results
    return [token+"_POS-"+tag for token,
            tag in nltk.pos_tag(tokens) if tag.startswith('NN')]


# Using NLTK lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class LengthFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def _get_features(self, doc):
        return {"words": len(doc),
                "unique_words": len(set(doc)) }

    def transform(self, raw_documents):
        return [ self._get_features(doc) for doc in raw_documents]


# Show Distribution of Data
def distribution(trainClass, testClass):

    labels = ["books", "camera", "dvd", "health", "music", "software"]
    count_training = [0, 0, 0, 0, 0, 0]
    count_testing = [0, 0, 0, 0, 0, 0]

    i = 0
    for label in labels:
        for cls in trainClass:
            if cls == label:
                count_training[i] += 1
        i += 1

    i = 0
    for label in labels:
        for cls in testClass:
            if cls == label:
                count_testing[i] += 1
        i += 1

    print("Distribution of classes in Training Set:")
    print(labels)
    print(count_training)

    print("\nDistribution of classes in Testing Set:")
    print(labels)
    print(count_testing)


# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf, stopwords_bn):
    # let's use the

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer( preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))
    else:
        vec = CountVectorizer( preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))

    return vec


# Modified TF-IDF vectorization for features: Uses pre-processing
# based on the value of tfidf (True/False)
def tf_idf_func_modified_for_Topic(tfidf, stopwords_bn):

    # if tfidf:
    #     vec = TfidfVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = identity)
    # else:
    #     vec = CountVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = tokenize_pos)
    #
    if tfidf:
        vec = TfidfVectorizer( preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))
    else:
        vec = CountVectorizer( preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))

    return vec


# Modified TF-IDF vectorization for features: Uses pre-processing
# based on the value of tfidf (True/False)
def tf_idf_func_modified_for_Sentiment(tfidf, stopwords_bn):

    if tfidf:
        vec = TfidfVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = identity)
    else:
        vec = CountVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = identity, ngram_range=(2, 2))


    # # Using Length Vectorizer combined with Tf-Idf and Count Vectorizer (**warning it will take so much time for Linear SVM)
    # tfidf_vec = TfidfVectorizer(stop_words='english', preprocessor = identity,
    #                            tokenizer = tokenize_pos, ngram_range=(2, 2))
    #
    # count_vec = CountVectorizer(analyzer=identity, stop_words='english',
    #                            preprocessor = identity, tokenizer = LemmaTokenizer, ngram_range=(2, 2))
    #
    # length_vec = Pipeline([
    #                 ('textstats', LengthFeatures()),
    #                 ('vec', DictVectorizer())
    #             ])
    #
    # # Here we are taking all 3 the above vectorizer (you can remove 1 or 2) - Usually tf-idf works best
    # vec = FeatureUnion([("tfidf", tfidf_vec), ("count", count_vec), ('textstats', length_vec)])

    return vec


# K-Means: Exercise - 3.2.1 - Six way classification
# The value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def KMeans_Clustering(docs, labels, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    # if you do no want to apply any pre-processor just use tf_idf_func()
    if use_sentiment:
        vec = tf_idf_func_modified_for_Sentiment(tfIdf, stopwords_bn)
    else:
        vec = tf_idf_func_modified_for_Topic(tfIdf, stopwords_bn)
        # vec = tf_idf_func(tfIdf)

    # performing vectorization (because we can't use pipeline here)
    X = vec.fit_transform(docs)

    # this is to get the number of class/labels
    true_k = numpy.unique(labels).shape[0]
    # print("\nNo. of Labels = ", true_k, "\n")

    # ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
    # verbose=1 shows the iteration (0 to hide them)
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10, verbose=0)

    t0 = time.time()
    # Fit/Train Multinomial Naive Bayes km according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    km.fit(X)

    cluster_time = time.time() - t0

    # Use the km to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess although it is not necessary it could be found in km.labels_
    testGuess = km.labels_

    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### K-Means Clustering For ", true_k, " way ", classType, " (", tfIDF_type, ") ###########")

    # Call to function(s) to do the jobs ^_^
    calculate_measures(km, labels, testGuess)

    print("\nClustering Time: ", cluster_time)
    print("\n")


def KMeans_Clustering_loop(docs, labels, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier

    n_init_val = []
    adjtd_rand = []
    v_score = []

    # performing vectorization over the documents (because we can't use pipeline here)
    X = vec.fit_transform(docs)

    true_k = numpy.unique(labels).shape[0]

    print("\n##### Output of K-Means Clustering for different values of n_init (1-15) [TfidfVectorizer] #####")

    # n_init = n Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    for n in range(1, 16):
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=n, verbose=0)

        km.fit(X)

        n_init_val.append(n)
        adjtd_rand.append(adjusted_rand_score(labels, km.labels_))
        v_score.append(v_measure_score(labels, km.labels_))

        print("n_init=",n_init_val[n-1],"   Adjusted Rand-Index=",adjtd_rand[n-1],"     V-measure=",v_score[n-1])

    # print()
    # for i in range(1, 16):
    #     print("n_init=",n_init_val[i-1],"   Adjusted Rand-Index=",adjtd_rand[i-1],"     V-measure=",v_score[i-1])

    return n_init_val, adjtd_rand, v_score


# for calculating different scores
def calculate_measures(km, labels, testGuess):

    # km.labels_ is actually the testGuess - so we don't need to explicitly define it...
    # print(homogeneity_completeness_v_measure(labels, testGuess))

    print("\nHomogeneity = %0.3f" % homogeneity_score(labels, km.labels_))
    print("Completeness = %0.3f" % completeness_score(labels, km.labels_))
    print("V-measure = %0.3f" % v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index = %.3f" % adjusted_rand_score(labels, km.labels_))


# Draw plots based on different values of some parameter (val)
def draw_plots(n_init_val, adjtd_rand, v_score, value_name):
    plt.plot(n_init_val, adjtd_rand, color='red', label='Adjusted Rand-Index')
    plt.plot(n_init_val, v_score, color='yellow', label='V-measure')

    x_label = "Values of "+value_name
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.legend()

    plt.show()


# This function runs Naive Bayes, Decision Tree and K-NN classifiers
def run_all_classifiers(docs, labels, stopwords_bn, use_sentiment):

    # KM - (True for Binary Class) with Tf-Idf Vectorizer
    KMeans_Clustering(docs, labels, stopwords_bn, True, use_sentiment)

    # KM - (True for Binary Class) with Count Vectorizer
    KMeans_Clustering(docs, labels, stopwords_bn, False, use_sentiment)

    # Try different values of n_init in K-Means clustering and To collect the data for curve
    n_init_val, adjtd_rand, v_score = KMeans_Clustering_loop(docs, labels, stopwords_bn, True, False)
    draw_plots(n_init_val, adjtd_rand, v_score, "n_init")


# this is the main function but you can name it anyway you want
def main():

    print("Wait for it... Don't panic (Porter's Stemmer is taking time...)\n")

    # Reads files for K-Means clustering - Exercise 3.2.1 Six way classification
    documents, labels_topic, labels_sentiment = read_corpus('trainset.txt')

    # Data of only Two labels "camera" and "books" - 3.2.2 Binary classification and feature selection
    docs_two, labels_twoTopic, labels_twoSenti = separate_two_labels(documents, labels_topic, labels_sentiment, "camera", "books")

    # show the distribution of classes in training and testing set
    # distribution(trainClass, testClass)

    # Run All Models for Six-Way TopicType and for the whole Dataset (use_sentiment = False)
    run_all_classifiers(documents, labels_topic, False)

    # Run All Models for Two Labels ("camera" & "books") and for Sentiment Type of those Two Labels
    run_all_classifiers(docs_two, labels_twoSenti, True)                    # without stemmer
    run_all_classifiers(stem_documents(docs_two), labels_twoTopic, False)   # with stemmer


# program starts from here
if __name__ == '__main__':
    main()
