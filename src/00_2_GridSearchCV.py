#!/usr/local/bin/python
# -*- coding: utf-8 -*
from __future__ import print_function

import sys
from pymongo import MongoClient
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
  
from sklearn import datasets  
from sklearn.cross_validation import train_test_split  
from sklearn.grid_search import GridSearchCV  
from sklearn.metrics import classification_report  
from sklearn.svm import SVC 
import numpy


print("start...")

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
segs = db.segments
t_segs = db.test_sengments
features = db.features
# tmp=db.tmp

# varible
X_train = []
X_test = []
y_train = []
y_test = []
y_pred = []
feature = []

# load vectorizer
vectorizer_train = CountVectorizer(min_df=1)
vectorizer_test = CountVectorizer(min_df=1)

# load features
fcursor = features.find()
for fcsr in fcursor:
    feature = fcsr["features"]
fture = " ".join(feature)
i = 0
# calculate tf and load X and y
# X_train y_train
cursor = segs.find()
for csr in cursor:
    i = i + 1
    if i % 10000 == 0:
        print("load train num %d"%i)
    corpus_tmp = []
    # append judgement to y
    judge = csr["judgement"]
    y_train.append(int(judge))

    # calculate tf
    words = csr["segment"]
    str = ""
    for word in words:
        if word in feature:
            str = str + word + " "
    corpus_tmp.append(fture)
    corpus_tmp.append(str)
    xx_train = vectorizer_train.fit_transform(corpus_tmp)
    X_train.append(xx_train.toarray()[1])
	
# X_test y_test
i = 0
cursor_test = t_segs.find()
for csr in cursor_test:
    i = i + 1
    if i % 1000 == 0:
        print("load test num %d"%i)
    corpus_tmp = []
    # append judgement to y
    judge = csr["judgement"]
    y_test.append(int(judge))

    # calculate tf
    words = csr["segment"]
    str = ""
    for word in words:
        if word in feature:
            str = str + word + " "
    corpus_tmp.append(fture)
    corpus_tmp.append(str)
    xx_test = vectorizer_test.fit_transform(corpus_tmp)
    X_test.append(xx_test.toarray()[1])
print("data load over.")
'''
# examples
X_train = numpy.array([[1,1,1,1,1],[0,1,0,1,0],[0,0,0,0,0],[1,0,1,0,1],[1,1,0,0,0]])
X_test = numpy.array([[1,1,1,1,1],[0,1,0,1,0]])
y_train = numpy.array([1,0,1,0,1])
y_test = numpy.array([1,0])
'''
# Set the parameters by cross-validation  
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,2e-3],  
                     'C': [1.6]}
					  #,{'kernel': ['linear'], 'C': [1, 10, 100]}
					 ]  
  
scores = ['precision', 'recall']  
  
for score in scores:  
    print("# Tuning hyper-parameters for %s" % score)  
    print()  
  
    clf = GridSearchCV(SVC(C=1), tuned_parameters, # cv=5,  
                       scoring='%s_weighted' % score)  
    clf.fit(X_train, y_train)  
  
    print("Best parameters set found on development set:")  
    print()  
    print(clf.best_params_)  
    print()  
    print("Grid scores on development set:")  
    print()  
    for params, mean_score, scores in clf.grid_scores_:  
        print("%0.3f (+/-%0.03f) for %r"  
              % (mean_score, scores.std() * 2, params))  
    print()  
  
    print("Detailed classification report:")  
    print()  
    print("The model is trained on the full development set.")  
    print("The scores are computed on the full evaluation set.")  
    print()  
    y_true = y_test 
    y_pred = clf.predict(X_test)
    print(classification_report(numpy.array(y_true), numpy.array(y_pred)))  
    print()  
