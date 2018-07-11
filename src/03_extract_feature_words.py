#!/usr/local/bin/python
# -*- coding: utf-8 -*

import sys
import re
import os

path = os.path.abspath('..')  # 工程根目录

sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser
from pymongo import MongoClient

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.sa_train
allwords = db.words
features = db.features

# load local library
jieba.analyse.set_stop_words(path + r"\dict\stopLibrary.dic")
jieba.load_userdict(path + r"\dict\user_dict.txt")

# input train set
filename = path + r"\data\train\total_train.txt"

# extract feature
content = open(filename, 'r', encoding='utf-8').read()
wordcnt = 0

cursor = allwords.find()
for csr in cursor:
    words = csr["words"]
    wordcnt = len(words)
    print(wordcnt)
topK = int(wordcnt * 0.2)
print(topK)
tags = jieba.analyse.extract_tags(content, topK=topK)
features.insert({"features": tags})
conn.close()
