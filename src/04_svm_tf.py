#!/usr/local/bin/python
# -*- coding: utf-8 -*

import sys,os
from pymongo import MongoClient
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

path = os.path.abspath('..')

print("start...")
# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
segs = db.segments
features = db.features_idf
ford = db.features_order

feature_num = 500

# varible
X = []
y = []
feature = []
idf = []
fture = ""

# load vectorizer
vectorizer = CountVectorizer(min_df=1)

# load features & idf
fcursor = features.find()
fcursor = fcursor[0]["features"]
for i in range(feature_num):
    fcsr = fcursor[str(i)]
    fea = fcsr["feature"]
    if i != 0 :
        fture = fture + " "
    fture = fture + str(fea)
    feature.append(fea)
    idf.append(fcsr["idf"])
print("load feature over.")

i = 0

# calculate tf and load X and y
cursor = segs.find()
for csr in cursor:
    i = i + 1
    if i % 250 == 0:
        print("load %d data"%i)
    corpus_tmp = []
    # append judgement to y
    judge = csr["judgement"]
    y.append(int(judge))

    # calculate tf
    words = csr["segment"]
    str = ""
    for word in words:
        if word in feature:
            str = str + word + " "
    corpus_tmp.append(fture)
    corpus_tmp.append(str)
    xx = vectorizer.fit_transform(corpus_tmp)

    # calculate tf-idf
    xx = xx.toarray()[1]
    for j in range(feature_num):
        xx[j] = xx[j]*idf[j]

    X.append(xx)

# print xx.toarray()[0]
# print "\n"
# array=xx.toarray()[1]
# tmpa=[]
# for a in array:
#	tmpa.append(a)
# tmp.insert({"segment":words,"judgement":judge,"feature":vectorizer.get_feature_names(),"array":tmpa})
# tmp.insert({"y":y})
# X=vectorizer.fit_transform(corpus)

# tmp=",".join(fs)
# output.write(tmp)
# output.write(X.toarray())

# svm
print("train svm")
clf = svm.SVC(C= 3.2, gamma=0.002, kernel='rbf')
clf.fit(X, y)

print("save model")
joblib.dump(clf, path+"/model/model_tf_idf.m")
print("finish")
conn.close()
