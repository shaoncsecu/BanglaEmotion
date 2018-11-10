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

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import *
import numpy

# Go to the last two lines of this program to have an idea from start (bottom-up functional approach)

# Read the trainset.txt file and tokenize it.
# Append the text in documents array.
# Append one of the first two tokens (either sentiment type/topics type) in labels array depending on use_sentiment.

def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()                   # tokenize the lines

            documents.append(tokens[3:])                    # append the text - starts from 4th tokens

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )                  # tokens[1] is sentiment type (either pos/neg)
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )                  # tokens[0] is one of 6 topic types

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

#Made it a fuction so that we can call it multiple times (for both Binary and Multi-class)
def Multinomial_Naive_Bayes(trainDoc, trainClass, testDoc, testClass, stopwords_bn, tfIdf, use_sentiment):

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfIdf:
        vec = TfidfVectorizer(preprocessor = identity,
                              tokenizer = identity, ngram_range=(2, 2))
    else:
        vec = CountVectorizer(preprocessor = identity,
                              tokenizer = identity, ngram_range=(2, 2))

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', MultinomialNB())] )


    # Fit Multinomial Naive Bayes classifier according to Xtrain, Ytrain
    # Here Xtrain are the documents from training set and Ytrain is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    # Use the classifier to predict the class for all the documents in the test set Xtest
    # Save those output class labels in Yguess
    testGuess = classifier.predict(testDoc)

    # Just to know the output type
    classType = "Topic Class"
    if use_sentiment:
        classType = "Sentiment Class"

    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"     # This is ternary conditional operator in python

    print("\n########### Naive Bayes Classifier For ", classType, " (", tfIDF_type, ") ###########")


    title = "Naive Bayes Classifier ("+tfIDF_type+")"
    # Call to function(s) to do the jobs ^_^
    calculate_measures(classifier, testClass, testGuess, title)

    # calculate_probabilities(classifier, testClass, trainClass)


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



# Exercise 1.4 – Probabilities
def calculate_probabilities(classifier, Xtest, Ytrain):

    # Posterior probabilities for every documents ()
    print("Posterior probabilities:")
    print(classifier.classes_)
    print(classifier.predict_proba(Xtest))      # Posterior probability depends on the documents in Test Set(Xtest)

    # Prior Probability for classes
    prior = classifier.predict_proba(Ytrain)     # Prior probability depends on the occurrence of class in Training Set(Ytrain)
    finalPrior = prior[len(prior)-1:]           # Last row in the array is the final prior probability (as it builds up gradually: N(class i)/N(doc))

    print("\nPrior Probability(Probability of Class):")
    print(classifier.classes_)
    print(finalPrior)


# This function runs Naive Bayes
def run_all_classifiers(trainDoc, trainClass, testDoc, testClass, stopwords_bn, use_sentiment):

    # NB - with Tf-Idf Vectorizer
    Multinomial_Naive_Bayes(trainDoc, trainClass, testDoc, testClass, stopwords_bn, True, use_sentiment)

    # NB - with Count Vectorizer
    Multinomial_Naive_Bayes(trainDoc, trainClass, testDoc, testClass, stopwords_bn, False, use_sentiment)

