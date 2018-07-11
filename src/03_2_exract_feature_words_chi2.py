#!/usr/local/bin/python
# -*- coding: utf-8 -*

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import jieba
import re
import os

path = os.path.abspath('..')  # 工程根目录

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
allwords = db.words_chi2
features = db.features

# load directory
jieba.load_userdict(path + r"\dict\user_dict.txt")

# filename
filename1 = path + r'\data\train\pos_train.txt'
filename2 = path + r'\data\train\neg_train.txt'

# load file into list
corpus0 = []
y = []
with open(filename1, encoding='utf-8') as reader:   # filename1 全部评论
        for index, line in enumerate(reader):
            if index < 10000:     # 前10000写入训练用
                re.sub(r'[A-Za-z0-9.。,，\\]', "", line)
                corpus0.append(line.replace('\n', ''))
                y.append(0)
        reader.close()

with open(filename2, encoding='utf-8') as reader:   # filename1 全部评论
        for index, line in enumerate(reader):
            if index < 10000:     # 前10000写入训练用
                re.sub(r'[A-Za-z0-9.。,，\\]', "", line)
                corpus0.append(line.replace('\n', ''))
                y.append(1)
        reader.close()

corpus = []
for c in corpus0:
    corpus.append(' '.join(jieba.cut(c)))

vectorizer = CountVectorizer()  # 转为词频矩阵
X = vectorizer.fit_transform(corpus)
allwords.insert({"_id": 1, "words": list(set(vectorizer.get_feature_names()))})
selector = SelectKBest(chi2, k=500)
selector.fit_transform(X, y)
idxs_selected = selector.get_support(indices=True)   # 返回索引
# idxs_selected_1 = selector.get_support(indices=False)   # 返回布尔值
# print(idxs_selected)
# print(idxs_selected_1)
feature_list = list(vectorizer.get_feature_names())

list1 = []
for i in range(len(feature_list)):
    if i in idxs_selected:
        list1.append(feature_list[i])
features.insert({"features": list(set(list1))})