#!/usr/local/bin/python

from __future__ import division

from pymongo import MongoClient
from sklearn.externals import joblib
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
import os

path = os.path.abspath('..')  # 工程根目录

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
sres = db.test_result_rbf3_2
segs = db.test_segments
features = db.features

# load svm
print("load model")
clf = joblib.load(path + r"\model\svm_train_model.m")

# load local library
print("load local_dict & stop_library")
jieba.load_userdict(path + r"\dict\user_dict.txt")
stopkey = [line.strip() for line in open(path + r'\dict\stopLibrary.dic'
                                         , encoding='utf-8').readlines()]
stopkey.append("")
stoplabel = ["x"]

# input test set
pos_in = open(path + r"\data\train\pos_test.txt", encoding='utf-8')
neg_in = open(path + r"\data\train\neg_test.txt", encoding='utf-8')

# read pos comments and seprate words
print("seprate test")
for line in pos_in:
    words = pseg.cut(line)
    segment = []
    for word in words:
        if (word.word not in stopkey) and (word.flag not in stoplabel):
            tmp = word.word
            segment.append(tmp)
            # mongo output_2
            # segs.insert_one({"judgement": "0", "comment": line, "segment": segment})

# read neg comments and seprate words
for line in neg_in:
    words = pseg.cut(line)
    segment = []
    for word in words:
        if (word.word not in stopkey) and (word.flag not in stoplabel):
            tmp = word.word
            segment.append(tmp)
            # mongo output_2
            # segs.insert_one({"judgement": "1", "comment": line, "segment": segment})

# varible
X = []
y = []
feature = []
# load vectorizer
vectorizer = CountVectorizer(min_df=1)
# load features
fcursor = features.find()
for fcsr in fcursor:
    feature = fcsr["features"]
fture = " ".join(feature)
i = 0

# calculate tf and load X and y
print("clf running")
cursor = segs.find()
sum = 0
for csr in cursor:
    i = i + 1
    if i % 200 == 0:
        print(i)
    corpus_tmp = []
    # get judgement & comment
    judge = csr["judgement"]
    comment = csr["comment"]
    # calculate tf
    words = csr["segment"]
    str1 = ""
    for word in words:
        if word in feature:
            str1 = str1 + word + " "
    corpus_tmp.append(fture)
    corpus_tmp.append(str1)
    xx = vectorizer.fit_transform(corpus_tmp)
    X.append(xx.toarray()[1])
    y = clf.predict(X)
    if np.int32(y[0]) != np.int32(judge):
        sum += 1
        sres.insert({"comment": comment, "judgement": judge, "clf_judge": str(y[0])})
    X.clear()
print("sum:" + str(sum))
print("count:" + str(i))
rate = sum / i
print("rate:" + str(rate))

conn.close()
