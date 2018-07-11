#!/usr/local/bin/python
# -*- coding: utf-8 -*

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import jieba
import re,os,math

path = os.path.abspath('..')

print("start...")
# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
segs = db.segments
features = db.features
features_idf = db.features_idf
# tmp=db.tmp

# varible
train_num = 8000*2
X = []
feature = []
fture = ""

# load vectorizer
vectorizer = CountVectorizer(min_df=1)

# load features
fcursor = features.find()
for fcsr in fcursor:
    feature = fcsr["features"]
    fture = " ".join(feature)
print(fture)

i = 0

# calculate tf and load X and y
cursor = segs.find()
for csr in cursor:
    corpus_tmp = []
    # calculate tf
    words = csr["segment"]
    com = ""
    for word in words:
        if word in feature:
            com = com + word + " "
    corpus_tmp.append(fture)
    corpus_tmp.append(com)
    xx = vectorizer.fit_transform(corpus_tmp)
    X.append(xx.toarray()[1])

feature_names = vectorizer.get_feature_names()
result = [0 for i in range(500)]
for x in X:
    result = result+x

idf = {}
for i in range(500):
    if result[i]>0:
        feature_idf = math.log(train_num/(result[i]+1))
    if result[i]==0:
        feature_idf = 0
    idf[str(i)] = {"feature":feature_names[i], "idf":feature_idf}

features_idf.insert({"features":idf})