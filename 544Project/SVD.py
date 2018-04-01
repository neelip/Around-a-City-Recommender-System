# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:28:08 2017

@author: neelu
"""
import numpy as np
import csv 
import nltk
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import logging
from random import randint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


users = list()
venues = list()
comments = list()


lda = models.LdaModel.load('model.lda')

with open("./training.csv", "r") as trainFile:
    reader = csv.reader(trainFile)
    
    for row in reader:
        if len(row) != 0:
            users.append(int (row[0]))
            venues.append(int (row[1]))
            comments.append((row[2]))
            
            
core_set = {}
for i in range(len(users)):
    if users[i] not in core_set:
        core_set[users[i]] = {}
        core_set[users[i]][venues[i]] = comments[i]
    else :
        if venues[i] not in core_set[users[i]]:
            core_set[users[i]][venues[i]] = comments[i]
        else :
            core_set[users[i]][venues[i]] = core_set[users[i]][venues[i]]+(comments[i])
        
corpus_pre = list()
print (core_set[22278])
for k, v in core_set.items():
    temp = list()
    for i in v.values():
        
        temp.append(i)
    corpus_pre.append(temp)
    
user_index = randint(0,len(users))
user_key = users[user_index]
user_part1 = list()
for k,v in core_set[user_key].items():
    user_part1.append(v)
    
            
stopWords = set(stopwords.words('english'))

user_part2 = [[word.lower() for word in comment.lower().split() if word.lower() not in stopWords and word.isalpha()] for comment in user_part1]
user_dictionary = corpora.Dictionary(user_part2)
user_corpus = [user_dictionary.doc2bow(word) for word in user_part2]

corpus_lda = lda[user_corpus]

num_topics = 2
a = [0.0] * num_topics

for doc in corpus_lda:
    for id, score in doc:
        a[id] = a[id]+score;

for i in range(len(a)):
    a[i] = a[i]/len(user_corpus)

