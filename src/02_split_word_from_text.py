#!/usr/local/bin/python
# -*- coding: utf-8 -*

# split words from pos and neg text train set
# insert results into sa_train
# seg result of each text
# total kind of words of whole train set

import sys
import jieba
import jieba.posseg as pseg
from pymongo import MongoClient
import os

path = os.path.abspath('..')  # 工程根目录

# load user dict to jieba
jieba.load_userdict(path + '/dict/user_dict.txt')
stopkey = [line.strip() for line in open(path + '/dict/stopLibrary.dic', encoding='utf-8').readlines()]
stopkey.append("")
stoplabel = ["x"]

# input train set
pos_in = open(path + "/data/train/pos_train.txt", encoding='utf-8')
neg_in = open(path + "/data/train/neg_train.txt", encoding='utf-8')

# local fle to save results
pos_out = open(path + "/data/train/pos_train_segment.txt", "w+", encoding='utf-8')
neg_out = open(path + "/data/train/neg_train_segment.txt", "w+", encoding='utf-8')

# connect to mongodb
conn = MongoClient(localhost, 27017)
db = conn.jdcomment
allwords = db.words
segs = db.segments
allwords.insert_one({"_id": "1"})

# read pos comments and seprate words
for line in pos_in:
    words = pseg.cut(line)
    segment = []
    for word in words:
        if (word.word not in stopkey) and (word.flag not in stoplabel):
            tmp = word.word
            segment.append(tmp)
            # mongo output_1
            allwords.update({"_id": "1"}, {"$addToSet": {"words": tmp}})

        # mongo output_2
    segs.insert_one({"judgement": "0", "comment": line, "segment": segment})

    # file output
    segstr = ", ".join(segment)
    pos_out.write(line + "	" + str(segstr) + "\n")

# read neg comments and seprate words
for line in neg_in:
    words = pseg.cut(line)
    segment = []
    for word in words:
        if (word.word not in stopkey) and (word.flag not in stoplabel):
            tmp = word.word
            segment.append(tmp)
            # mongo output_1
            allwords.update({"_id": "1"}, {"$addToSet": {"words": tmp}})
        # mongo output_2
    segs.insert_one({"judgement": "1", "comment": line, "segment": segment})

    # file output
    segstr = ", ".join(segment)
    neg_out.write(line + "    " + str(segstr) + "\n")

conn.close()
pos_in.close()
neg_in.close()
pos_out.close()
neg_out.close()
