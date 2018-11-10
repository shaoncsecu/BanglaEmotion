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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy
from sklearn.metrics import *
import time
import sys

# Go to the last two lines of this program to have an idea from start (bottom-up functional approach)


# reads the two files as command line argument
# Example: DT_KNN.py <trainset> <testset>
def read_files():
    with open(sys.argv[1], 'r', encoding='utf-8') as train:
        trainData = train.readlines()   # copy the content of the file in a list

    with open(sys.argv[2], 'r', encoding='utf-8') as test:
        testData = test.readlines()

    return trainData, testData


# we are using NLTK stemmer to stem multiple words into root
def apply_stemmer(doc):
    stemmer = PorterStemmer()

    roots = [stemmer.stem(plural) for plural in doc]

    return roots

# Tokenize and Append the text in documents array.
# Append one of the first two tokens (either sentiment type (true)/topics type (false)) in labels array depending on use_sentiment.
def modify_corpus(data, use_sentiment):

    documents = []
    labels = []

    for line in data:
        tokens = line.strip().split()  # tokenize the lines

        documents.append(tokens[3:])  # append the text - starts from 4th tokens

        if use_sentiment:
            # 2-class problem: positive vs negative
            labels.append(tokens[1])  # tokens[1] is sentiment type (either pos/neg)
        else:
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens[0])  # tokens[0] is one of 6 topic types

    stemmed_documents = []
    for doc in documents:
        stemmed_documents.append(apply_stemmer(doc))

    return stemmed_documents, labels


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


# a dummy function that just returns its input
def identity(x):
    return x


# Using NLTK lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf, stopwords_bn):
    # let's use the

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    # if tfidf:
    #     vec = TfidfVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))
    # else:
    #     vec = CountVectorizer(stop_words=stopwords_bn, preprocessor = identity, tokenizer = identity, ngram_range=(1, 3))

    # using lemmatizer doesn't improve performance
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec


# Naive Bayes classifier: the value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def NB_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', MultinomialNB())] )

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

    print("\n########### Naive Bayes Classifier For ", classType, " (", tfIDF_type, ") ###########")

    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess)

    # Showing 10 fold cross validation score cv = no. of folds
    # print("Cross Validation:\n", cross_val_score(classifier, testDoc, testClass, cv=10))
    print()
    print("Training Time: ", train_time)
    print("Testing Time: ", test_time)

    calculate_probabilities(classifier, testClass, trainClass)

# Exercise 2.1.2 – Decision Tree
# Decision Trees classifier: the value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def Decision_Trees(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', DecisionTreeClassifier())] )

    # Try to run the above classifier with the following parameters and see performance change
    # DecisionTreeClassifier(min_samples_split=3, min_samples_leaf=2, max_depth=10, max_features=1000)

    t0 = time.time()

    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0

    t1 = time.time()

    testGuess = classifier.predict(testDoc)

    test_time = time.time() - t1


    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### Decision Trees Classifier For "+classType+" (", tfIDF_type, ") ###########")

    title = "Decision Tree Classifier ("+tfIDF_type+")"
    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess, title)

    # print("Cross Validation:\n", cross_val_score(classifier, testDoc, testClass, cv=10))
    print()
    print("Training Time: ", train_time)
    print("Testing Time: ", test_time)


# Exercise 2.2 – K-Nearest Neighbor
# K-Nearest Neighbor classifier: the value of boolean arg - use_sentiment decides on binary (True - sentiment) vs multi-class (False - Topic) classification
def KNN_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', KNeighborsClassifier(n_neighbors=3))] )

    t0 = time.time()

    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0

    t1 = time.time()

    testGuess = classifier.predict(testDoc)

    test_time = time.time() - t1

    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### K-Nearest Neighbor Classifier For "+classType+" (", tfIDF_type, ") ###########")

    title = "KNN Classifier ("+tfIDF_type+")"
    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess, title)

    # print("Cross Validation:\n", cross_val_score(classifier, testDoc, testClass, cv=10))
    print()
    print("Training Time: ", train_time)
    print("Testing Time: ", test_time)


# Exercise 2.2.1 – K-Nearest Neighbor (for different accuracy and f1-scores)
# K-Nearest Neighbor classifiers results for different values of K
def KNN_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf, stopwords_bn)

    # combine the vectorizer with a Decision Trees classifier

    k_val = []
    accu = []
    f1 = []

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n##### Output of K-NN classifier for different values of K (1-20) (", tfIDF_type, ") #####\n")

    for k in range(1, 31):
        classifier = Pipeline( [('vec', vec),
                                ('cls', KNeighborsClassifier(n_neighbors=k))] )

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        k_val.append(k)
        accu.append(accuracy_score(testClass, testGuess))
        f1.append(f1_score(testClass, testGuess, average='macro'))

        # print("K =", k, ": Accuracy = ", round(accuracy_score(testClass, testGuess), 3), "  F1-score (micro) = ", round(f1_score(testClass, testGuess, average='macro'), 3))

    print()
    for i in range(1, 30):
        print("K=",k_val[i],"   Accuracy=",round(accu[i], 3),"     F1-score=",round(f1[i], 3))

    return k_val, accu, f1


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


def draw_plots(k_val, accu, f1):
    plt.plot(k_val, accu, color='red', label='Accuracy')
    plt.plot(k_val, f1, color='yellow', label='F1-score')
    plt.xlabel('Values of K')
    plt.legend()

    plt.show()


# This function runs Naive Bayes, Decision Tree and K-NN classifiers
def run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment):

    # Test the Naive Bayes (False for Topic Class) with Tf-Idf vectorizer
    #NB_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Test the Naive Bayes (False for Topic Class) with CountVectorizer
    # NB_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)

    # Test the Decision_Trees (False for Topic Class) with Tf-Idf vectorizer
    Decision_Trees(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Test the Decision_Trees (False for Topic Class) with CountVectorizer
    Decision_Trees(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)

    # Test the KNN classfier (False for Topic Class) with Tf-Idf vectorizer
    KNN_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # Test the KNN classfier (False for Topic Class) with CountVectorizer
    KNN_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)

    #To collect the data for curve
    k_val, accu, f1 = KNN_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)
    draw_plots(k_val, accu, f1)

    #To collect the data for curve
    k_val, accu, f1 = KNN_loop(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)
    draw_plots(k_val, accu, f1)


# This function runs Naive Bayes with Tf-Idf Vectorizers and some Pre-preprocessing
def run_best_model(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment):

    # Test the Naive Bayes (False for Topic Class) with Tf-Idf vectorizer
    NB_classifier(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)


# this is the main function but you can name it anyway you want
def main():

    print("Wait for it... Don't panic (Porter's Stemmer is taking time...)\n")

    # reads files <trainSet> <testSet> as command line argument
    trainSet, testSet = read_files()

    # divides the files into tokenized documents and class labels (A False means the 6 Topic Type Classification)
    trainDoc, trainClass = modify_corpus(trainSet, False)
    testDoc, testClass = modify_corpus(testSet, False)

    # show the distribution of classes in training and testing set
    # distribution(trainClass, testClass)

    # Running the best model based among 3 (if you want to see the output of every model then uncomment the above function)
    run_best_model(trainDoc, trainClass, testDoc, testClass)

    print("\n\n Do you want to See the Output of other classifiers(Decsision Tree/K-NN) too?:")

    c = str(input("[Y/N]:"))

    if c =='Y' or c == 'y':
        # run all the 3 classifiers
        run_all_classifiers(trainDoc, trainClass, testDoc, testClass)


# program starts from here
if __name__ == '__main__':
    main()
