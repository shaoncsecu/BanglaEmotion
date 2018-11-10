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
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import *
import numpy
import warnings
import time
import sys
import K_Means
import Naive_Bayes
import DT_KNN

# Filters all warnings - specially to ignore warnings due to 0.0 in precision and recall and f-measures
warnings.filterwarnings("ignore")

# Go to the last two lines of this program to have an idea from start (bottom-up functional approach)

# reads the two files as command line argument
# Example: DT_KNN.py <trainset> <testset>
def read_files():
    with open(sys.argv[1], 'r', encoding='utf-8') as train:
        trainData = train.readlines()   # copy the content of the file in a list

    with open(sys.argv[2], 'r', encoding='utf-8') as test:
        testData = test.readlines()

    # Read the Bangla Stop word list from file by: https://github.com/stopwords-iso/stopwords-bn
    with open('stopwords-bn.txt', 'r', encoding='utf-8') as test:
        stopwords_bn = test.readlines()
        # the above stopwords contains newline \n
        stop_bn = []

        for word in stopwords_bn:
            stop_bn.append(word.rstrip("\r\n"))

    return trainData, testData, stop_bn


# a dummy function that just returns its input
def identity(x):
    return x


# we are using NLTK stemmer to stem multiple words into root
def apply_stemmer(doc):
    stemmer = PorterStemmer()

    roots = [stemmer.stem(plural) for plural in doc]

    return roots


# NLTK POS Tagger
def tokenize_pos(tokens):
    return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


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


# Tokenize and Append the text in documents array.
# Append one of the first two tokens (either sentiment type (true)/topics type (false)) in labels array depending on use_sentiment.
def modify_corpus(data, use_sentiment):

    documents = []
    labels = []

    for line in data:
        tokens = line.strip().split()  # tokenize the lines

        documents.append(tokens[2:])  # append the text - starts from 3rd tokens

        if use_sentiment:
            # 2-class problem: positive vs negative
            labels.append(tokens[1])  # tokens[1] is sentiment type (either pos/neg)
        else:
            # 6-class problem: "sad", "happy", "disgust", "surprise", "fear", "angry"
            labels.append(tokens[0])  # tokens[0] is one of 6 topic types

    # Apply Porter's stemming (pre-processor)
    # stemmed_documents = []
    # for doc in documents:
    #     stemmed_documents.append(apply_stemmer(doc))
    #
    # return stemmed_documents, labels
    return documents, labels


# Show Distribution of Data
def distribution(trainClass, testClass):

    labels = ["sad", "happy", "disgust", "surprise", "fear", "angry"]
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
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec


# Modified TF-IDF vectorization for features: Uses pre-processing
# based on the value of tfidf (True/False)
def tf_idf_func_modified(tfidf, stopwords_bn):

    if tfidf:
        # Using only tfidf vectorizer
        tfidf_vec = TfidfVectorizer(preprocessor = identity,
                                    tokenizer = identity, ngram_range=(1, 3))
        return tfidf_vec

    else:
        # Using Length Vectorizer combined with Tf-Idf and Count Vectorizer (**warning it will take so much time for Linear SVM)
        tfidf_vec = TfidfVectorizer(stop_words=stopwords_bn, preprocessor = identity,
                                    tokenizer = identity, ngram_range=(1, 3))

        count_vec = CountVectorizer(analyzer=identity, stop_words=stopwords_bn,
                                   preprocessor = identity, tokenizer = LemmaTokenizer, ngram_range=(2, 2))

        length_vec = Pipeline([
                        ('textstats', LengthFeatures()),
                        ('vec', DictVectorizer())
                    ])

        # Here we are taking all 3 the above vectorizer (you can remove 1 or 2) - Usually tf-idf works best
        vec = FeatureUnion([("tfidf", tfidf_vec), ("count", count_vec), ('textstats', length_vec)])

        return vec

    # # With Stop_Words/Pos-Tagger/N-grams as features (Doesn't improve performance that much!!)
    # if tfidf:
    #     vec = TfidfVectorizer(stop_words='english', preprocessor = identity,
    #                           tokenizer = tokenize_pos, ngram_range=(2, 2))
    # else:
    #     vec = CountVectorizer(stop_words='english', preprocessor = identity,
    #                           tokenizer = tokenize_pos, ngram_range=(2, 2))

    # # Featuer Union Doesn't work that great
    # tfidf_vec = TfidfVectorizer(analyzer=identity, preprocessor = identity, tokenizer = identity)
    # count_vec = CountVectorizer(analyzer=identity, preprocessor = identity, tokenizer = identity, ngram_range=(2, 2))
    # vec = FeatureUnion([("count", count_vec), ("tfidf", tfidf_vec)])

    # # using lemmatizer doesn't improve performance
    # if tfidf:
    #     vec = TfidfVectorizer(analyzer=identity, stop_words='english',
    #                           preprocessor = identity, tokenizer = LemmaTokenizer)
    # else:
    #     vec = CountVectorizer(analyzer=identity, stop_words='english',
    #                           preprocessor = identity, tokenizer = LemmaTokenizer)

    # return vec


# 3.1.1 Default settings
# SVM Classifier: the value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def SVM_Normal(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    # if you do no want to apply any pre-processor just use tf_idf_func()
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='linear', C=1.0))] )

    t0 = time.time()
    # Fit/Train Multinomial Naive Bayes classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0

    t1 = time.time()
    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    test_time = time.time() - t1

    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### Default SVM Classifier For ", classType, " (", tfIDF_type, ") ###########")

    title = "Linear SVM (C = 1.0)"
    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess, title)

    # Doing cross validation on the Whole Data set
    cross_validation(classifier, trainDoc + testDoc, trainClass + testClass)

    print("\nTraining Time: ", train_time)
    print("Testing Time: ", test_time)


# 3.1.2 Setting C – (for different accuracy and f1-scores)
# SVM classifiers results for different values of C
def SVM_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier

    C_val = []
    accu = []
    f1 = []

    print("\n##### Output of SVM classifier for different values of C (1-10) [TfidfVectorizer] #####")
    c = 1

    for k in range(1, 11):
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='linear', C=c))] )       # An interval of 1

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        C_val.append(c)
        accu.append(accuracy_score(testClass, testGuess))
        f1.append(f1_score(testClass, testGuess, average='macro'))

        c += 1      # C in interval of 1
        # print("K =", k, ": Accuracy = ", round(accuracy_score(testClass, testGuess), 3), "  F1-score (micro) = ", round(f1_score(testClass, testGuess, average='macro'), 3))

    print()
    for i in range(1, 11):
        print("C=",round(C_val[i-1],1),"   Accuracy=",accu[i-1],"     F1-score=",f1[i-1])

    return C_val, accu, f1


# 3.1.3 Using a Non-Linear Kernel
# SVM Classifier: the value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    # if you do no want to apply any pre-processor just use tf_idf_func()
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='rbf', gamma=0.9, C=2.0))] )

    t0 = time.time()
    # Fit/Train Multinomial Naive Bayes classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0

    t1 = time.time()
    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    test_time = time.time() - t1

    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### Non-Linear SVM Classifier For ", classType, " (", tfIDF_type, ") ###########")

    title = "Non-Linear SVM [RBF kernel Gamma=0.9, C=2.0)]"
    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess, title)

    # Doing cross validation on the whole Dataset
    cross_validation(classifier, trainDoc + testDoc, trainClass + testClass)

    print("\nTraining Time: ", train_time)
    print("Testing Time: ", test_time)


def SVM_NonLinear_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func_modified(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier

    gamma_val = []
    accu = []
    f1 = []

    print("\n##### Output of Non-Linear SVM classifier for different values of Gamma (0.7-1.1) [TfidfVectorizer] #####")

    g = 0.1         # g for gamma starting with 0.7

    for c in range(1, 15):
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='rbf', gamma=g, C=2.0))] )

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        gamma_val.append(g)
        accu.append(accuracy_score(testClass, testGuess))
        f1.append(f1_score(testClass, testGuess, average='macro'))

        g += 0.1           # An interval of 0.1

        # print("K =", k, ": Accuracy = ", round(accuracy_score(testClass, testGuess), 3), "  F1-score (micro) = ", round(f1_score(testClass, testGuess, average='macro'), 3))

    print()
    for i in range(1, 15):
        print("Gamma=",round(gamma_val[i-1],1),"   Accuracy=",accu[i-1],"     F1-score=",f1[i-1])

    return gamma_val, accu, f1


# This function cross validates the Training set into 5 folds
# We will not use the test set to cross validate because we want to compare the results
# of average cross validation score to the normal scores with test set
def cross_validation(classifier, trainDoc, trainClass):

    # Showing 10 fold cross validation score cv = no. of folds
    n_fold = 5
    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold)

    print(n_fold,"-fold Cross Validation (Accuracy):\n", scores)
    print("\nAccuracy (Mean - Cross Validation): %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, scoring='f1_macro')

    print()
    print(n_fold,"-fold Cross Validation (f1-macro):\n", scores)
    print("\nF1-macro (Mean - Cross Validation): %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# for calculating different scores
def calculate_measures(classifier, testClass, testGuess, title):

    # Compare the accuracy of the output (Yguess) with the class labels of the original test set (Ytest)
    print("Accuracy = "+str(accuracy_score(testClass, testGuess)))
    print("F1-score(macro) = "+str(f1_score(testClass, testGuess, average='macro')))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(testClass, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    # Showing the Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(testClass, testGuess, labels=classifier.classes_)
    print(classifier.classes_)
    print(cm)
    print()

    # Drawing Confusion Matrix
    skplt.metrics.plot_confusion_matrix(testClass, testGuess, normalize=True)
    tick_marks = numpy.arange(len(classifier.classes_))
    plt.xticks(tick_marks, classifier.classes_, rotation=45)
    plt.yticks(tick_marks, classifier.classes_)
    plt.title("Normalized Confusion Matrix: "+title)
    plt.tight_layout()
    plt.show()


# Probabilities
def calculate_probabilities(classifier, testClass, trainClass):

    # Posterior probabilities for every documents ()
    print("\nPosterior probabilities:")
    print(classifier.classes_)
    print(classifier.predict_proba(testClass))      # Posterior probability depends on the documents in Test Set(Xtest)

    # Prior Probability for classes
    prior = classifier.predict_proba(trainClass)     # Prior probability depends on the occurrence of class in Training Set(Ytrain)
    finalPrior = prior[len(prior)-1:]           # Last row in the array is the final prior probability (as it builds up gradually: N(class i)/N(doc))

    print("\nPrior Probability(Probability of Class):")
    print(classifier.classes_)
    print(finalPrior)


# Draw plots based on different values of some parameter (val)
def draw_plots(val, accu, f1, value_name):
    plt.plot(val, accu, color='red', label='Accuracy')
    plt.plot(val, f1, color='yellow', label='F1-score')

    x_label = "Values of "+value_name
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.legend()

    plt.show()


# This function runs Naive Bayes, Decision Tree and K-NN classifiers
def run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment):

    # Test the SVM (True for Binary Class) with Tf-Idf Vectorizer
    SVM_Normal(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Test the SVM (True for Binary Class) with Count vectorizer
    # SVM_Normal(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Try different values of C in linear SVM and To collect the data for curve
    C_val, accu, f1 = SVM_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)
    draw_plots(C_val, accu, f1, "C")

    # Test the Non-Linear SVM (True for Binary Class) with Tf-Idf Vectorizer
    SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Test the Non-Linear SVM (True for Binary Class) with Count Vectorizer
    # SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)

    # Try different values of Gamma in Non-Linear SVM and To collect the data for curve
    gamma_val, accu, f1 = SVM_NonLinear_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)
    draw_plots(gamma_val, accu, f1, "Gamma")

# This function runs Naive Bayes with Tf-Idf Vectorizers and some Pre-preprocessing
def run_best_model(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment):

    # Test the SVM (True for Binary Class) with Tf-Idf Vectorizer
    # SVM_Normal(trainDoc, trainClass, testDoc, testClass, True, True)

    # Test the Non-Linear SVM (True for Binary Class) with Tf-Idf Vectorizer
    SVM_NonLinear(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)


# this is the main function but you can name it anyway you want
def main():

    # reads files <trainSet> <testSet> as command line argument
    trainSet, testSet, stopwords_bn = read_files()

    # True = Binary Sentiment Classification / False = Six Way Topic classification(multi-class label)
    use_sentiment = False

    # divides the files into tokenized documents and class labels
    # this is for SVM
    trainDoc, trainClass = modify_corpus(trainSet, use_sentiment)
    testDoc, testClass = modify_corpus(testSet, use_sentiment)

    # show the distribution of classes in training and testing set
    distribution(trainClass, testClass)

    # Running the best model based among 3 (if you want to see the output of every model then uncomment the above function)
    print("\n\n Running the best Model - Non-Linear SVM:")
    run_best_model(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment)

    # Run both linear and no-linear classifier with different C/Gamma values
    print("\n\n Do you want to See the Output of all variants of SVM classifier?:")
    c = str(input("[Y/N]:"))
    if c =='Y' or c == 'y':
        # run all the 3 classifiers
        run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment)


    # Run both linear and no-linear classifier with different C/Gamma values
    print("\n\n Do you want to run K-means clustering?:")
    c = str(input("[Y/N]:"))
    if c =='Y' or c == 'y':
        # calling K-Means Clustering doc, label, use_sentiment = False (sixway topic classification)
        K_Means.run_all_classifiers(trainDoc+testDoc, trainClass+testClass, stopwords_bn, use_sentiment)


    # Run Multinomial Naive Bayes Classifier
    print("\n\n Do you want to run  Multinomial Naive Bayes Classifier?:")
    c = str(input("[Y/N]:"))
    if c =='Y' or c == 'y':
        # calling NB classifier, use_sentiment = False (sixway topic classification)
        Naive_Bayes.run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment)


    # Run Multinomial Naive Bayes Classifier
    print("\n\n Do you want to run  Decisiton Tree/KNN Classifier?:")
    c = str(input("[Y/N]:"))
    if c =='Y' or c == 'y':
        # calling NB classifier, use_sentiment = False (sixway topic classification)
        DT_KNN.run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment)


# program starts from here
if __name__ == '__main__':
    main()
