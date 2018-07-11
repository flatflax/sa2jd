#!/usr/local/bin/python

from __future__ import division
from snownlp import SnowNLP
from pymongo import MongoClient
import pymongo
from sklearn.externals import joblib
import importlib
import sys

importlib.reload(sys)
sys.setdefaultencoding('utf-8')

# load svm
clf = joblib.load("/apps/youku/model/svm_train_model.m")

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.youku_pre
comnts = db.comments
sres = db.SVM_result

# step.0 read comments and judgement
cursor = comnts.find()
total = cursor.count()
print("total:" + str(total))
sum = 0
count = 0
for csr in cursor:
    comment = csr["comment"]
    judge = csr["judgement"]
    brand = csr["brand"]

score = SnowNLP(comment.decode('utf-8')).sentiments
count += 1
if score > 0.5 and judge == "1":
    sum += 1
    sres.insert({"comment": comment, "judgement": judge, "snow_judge": "0", "brand": brand})
if score <= 0.5 and judge == "0":
    sum += 1
    sres.insert({"comment": comment, "judgement": judge, "snow_judge": "1", "brand": brand})

print("sum:" + str(sum))
print("count:" + str(count))
rate = 1 - sum / count
print("rate:" + str(rate))

conn.close()
