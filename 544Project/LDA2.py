# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:20:24 2017

@author: neelima potharaj 
"""
import logging
import csv
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd
import math as mt
from sklearn.metrics.pairwise import cosine_similarity
import random
import scipy.stats as stats
from scipy.stats import linregress
import pylab as pl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import heapq
import operator
import sklearn.preprocessing as pp



def getTopic(lda, user_corpus, num_topics):
    
    user_lda = lda[user_corpus]

    a = [0.0] * num_topics

    for doc in user_lda:
        for id, score in doc:
            a[id] = a[id]+score;

    for i in range(len(a)):
        a[i] = a[i]/len(user_corpus)
        
    topic_id = a.index(max(a))
    
    
    
    return topic_id


def getLDAModel(corpus_tfidf, dictionary, num_topics):
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics= num_topics)
    
    return lda

def getTfidfModel(corpus):
    tfidf = models.TfidfModel(corpus)
    
    return tfidf

def prepareUser(index, trainingSet, tfidf):
    
    user_comments = tfidf[list(trainingSet[index].values())]
    
        
    return user_comments

def createMainCorpus(comments):
    
    
    
    stopWords = set(stopwords.words('english'))
    words = [[word.lower() for word in comment.lower().split() if word.lower() not in stopWords] for comment in comments ]
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(word) for word in words]
    
    return [dictionary, corpus]
    

def lists2Dict(users, venues, comments):
    core_set = {}
    for i in range(len(users)):
        if users[i] not in core_set:
            core_set[users[i]] = {}
            core_set[users[i]]["num_comments"] = 1
            core_set[users[i]][venues[i]] = comments[i]
        else :
            if venues[i] not in core_set[users[i]]:
                core_set[users[i]][venues[i]] = comments[i]
                core_set[users[i]]["num_comments"] = core_set[users[i]]["num_comments"] + 1;
            else :
                core_set[users[i]][venues[i]] = core_set[users[i]][venues[i]]+(comments[i])
                core_set[users[i]]["num_comments"] = core_set[users[i]]["num_comments"] + 1;
    return core_set

def graph(users, venues, modified):
    users = sorted(users)
    venues = sorted(venues)
    users = np.trim_zeros(users)
    venues = np.trim_zeros(venues)
    
    print ("Total")
    print (len(users))
    print (len(venues))
    print (np.min(users))
    print (np.min(venues))
    print (np.max(users))
    print (np.max(venues))
    print ("Mean")
    print (np.mean(users))
    print (np.mean(venues))
    
    
    
    fit1 = stats.norm.pdf(users, np.mean(users), np.std(users))
    
    pl.plot(users,fit1,'-o')
    
    pl.hist(users, normed = True, bins = 196)
    #pl.show()
    pl.xlabel('Number of Comments')
    pl.ylabel('Probability')
    
    
    if (modified == True):
        pl.title('Distribution of Comments - User Baed (Post-Modification)')
        pl.savefig("users_postModifying.png")
       
    else :
        pl.title('Distribution of Comments - User Baed (Pre-Modification)')
        pl.savefig("users_preModifying.png")
        
    
    pl.close()
    
    fit2 = stats.norm.pdf(venues, np.mean(venues), np.std(venues))
    
    pl.plot(venues, fit2, '-o')
    pl.hist(venues, normed = True, bins = 36)
    #pl.show()
    pl.xlabel('Number of Comments')
    pl.ylabel('Probability')
    
    
    if (modified == True):
        pl.title('Distribution of Comments - Venue Based (Post-Modification)')
        pl.savefig("venues_postModifying.png")
        
    else :
        pl.title('Distribution of Comments - Venue Based (Pre-Modification)')
        pl.savefig("venues_preModifying.png")
        
        
    pl.close()

def readFile():
    output = open('core_set_users.txt', 'rb')
    core_set_users = pickle.load(output)
    
    output2 = open('core_set_venues.txt', 'rb')
    core_set_venues = pickle.load(output2)
    
    output3 = open('comments.txt', 'rb')
    comments = pickle.load(output3)
    
    
    return [core_set_users, core_set_venues, comments]

def evaluate(testingSet, nearestNeighbors, user_id):
    
    matches = 0
    relevant = len(testingSet[user_id])
    
    print ('TestingSet')
    print (testingSet[user_id])
    
    for x in nearestNeighbors:
        
        
        
        if x in testingSet[user_id]:
            
            matches = matches + 1
    
    return [matches, relevant]

def removeDuplicates(trainingSet, recommended_venues, user_id):
    existing_venues = list(trainingSet[user_id].keys())
    print (existing_venues)
    recommended_venues = [x for x in recommended_venues if x not in existing_venues]

    return recommended_venues

def recommend(nearestNeighbors, trainingSet):
    
    recommended_venues = set()
    
    singles = list(trainingSet['Singles'].keys())
    
    for x in nearestNeighbors:
        if x in singles:
            
            recommended_venues.update(list(trainingSet['Singles'][x].keys()))
        else :
            if x != 3298 and x != 0:
                recommended_venues.update(list(trainingSet[x].keys()))
    
    return recommended_venues
    

def kNearest(user_id, similar_users, k):

    nearestNeighbors = np.array(similar_users)
    nearestNeighbors = nearestNeighbors.argsort()[-k:][::-1]
    
    print ("nearest")
    print (nearestNeighbors)

    return nearestNeighbors

def computeSimilarityMatrix(P_prime, num_users):
    sim_matrix = np.zeros(shape=(num_users+1, num_users+1))
    
    for x in range(1, num_users+1):
        
        for y in range(1, num_users+1):
            user = P_prime[x]
        
            user_B = P_prime[y]
        
        
            temp = linregress(user, user_B)
            sim_matrix[x, y]= temp.pvalue
            sim_matrix[y, x] = temp.pvalue
    
    return sim_matrix

def computeSimilarity(user_id, P_prime, num_users):
    
    similar_users = np.zeros(num_users+1, dtype='float')
    
    for x in range(1, len(similar_users)):
        
        user = P_prime[user_id]
        
        user_B = P_prime[x]
        
        
        temp = linregress(user, user_B)
        similar_users[x]= temp.pvalue
    
    return similar_users 

def computeSVD(P):
    
#    P_sparse = csc_matrix(P)
#    ut, s, vt = sparsesvd(P_sparse, 1000)
#    P_prime = np.dot(ut.T, np.dot(np.diag(s), vt))
    
#    
    U, S, V = np.linalg.svd(P, full_matrices=False)
    P_prime = np.dot(np.dot(U, np.diag(S)), V)
#    
#    
#    
    #print (np.std(P), np.std(P_prime), np.std(P-P_prime))
    #print (len(np.transpose(P_prime.nonzero())))
    
    return P_prime

def getPMatrix(trainingSet, topic_id, dictionary, num_users, num_venues, lda, tfidf):
    
    P_Matrix = np.zeros(shape = (num_users+1, num_venues+1))
     
    for k, v in trainingSet.items():
        
        if k == 'Singles':
            for x, y in v.items():
                venue_index = list(y.keys())[0]
                temp_comment = list(y.values())[0]
                temp_corpus = dictionary.doc2bow(temp_comment.lower().split())
                temp_lda = lda[temp_corpus]
                temp_theta = 0
                
                for doc1, doc2 in temp_lda:
                    if doc1 == topic_id:
                        temp_theta = float(doc2)
                        
                P_Matrix[x, venue_index] = temp_theta
        else :
            for x, y in v.items():
                temp_comment = y
                temp_corpus = dictionary.doc2bow(temp_comment.lower().split())
                temp_lda = lda[temp_corpus]
                temp_theta = 0
                
                for doc1, doc2 in temp_lda:
                    if doc1 == topic_id:
                        temp_theta = float(doc2)
                        
                        
                P_Matrix[k, x] = temp_theta
                        
    return P_Matrix


def getRecommendations(user_id, lda, trainingSet, k, dictionary, numbers, tfidf):
    
    num_topics = numbers['num_topics']
    num_users = numbers['num_users']
    num_venues = numbers['num_venues']
    
    user_comments = prepareUser(user_id, trainingSet, tfidf)
    user_corpus = [dictionary.doc2bow(comment.lower().split()) for comment in user_comments]
     
    #print (user_corpus)
    #print (user_id)
    
    topic_id = getTopic(lda, user_corpus, num_topics)
    
    
    P_matrix = getPMatrix(trainingSet, topic_id, dictionary, num_users, num_venues, lda)

    
   
    
    #print (np.count_nonzero(P_matrix))
    
    P_prime = computeSVD(P_matrix)
    

   
    similar_users = computeSimilarity(user_id, P_prime, num_users)
#    
#    
    nearestNeighbors = kNearest(user_id, similar_users, k)
    
    recommended_venues = recommend(nearestNeighbors, trainingSet)
##    
    return recommended_venues
#    

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat
        

def main_alt():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    [core_set_users, core_set_venues, comments] = readFile()
    

    
    users = [0 for x in range(len(core_set_users)+1)]
    venues = [0 for x in range(len(core_set_venues)+1)]
    
    num_users = len(core_set_users)
    num_venues = len(core_set_venues)
    
    print ("Valid Number of Tips: "+str(len(comments)))
    print ("Max Number of Tips by a User: "+str(len(users)-1))
    print ("Max Number of Tips for a Venues: "+str(len(venues)-1))
    
    maxTips_User = 0
    maxTips_Venue = 0
    
    trainingSet = dict()
    trainingSet['Singles'] = dict()
    testingSet = dict()

    
    sumOfComments = 0
    numUsers = 0
    
    for k, v in core_set_users.items():
        
        index = v['index']
        num_comments = v['num_comments']
        
        count = 0 
        halfset = 0
        if num_comments > 1:
            halfset = num_comments/2
            
        users[index] = num_comments
       
        sumOfComments = sumOfComments + num_comments
        if num_comments > maxTips_User:
            maxTips_User = num_comments
        
        for x, y in v.items():
            
            if isinstance(x, int):
                
                if num_comments <= 1:
                    trainingSet['Singles'][index] = dict()
                    trainingSet['Singles'][index][x] = y
                elif count < halfset and index not in testingSet:
                    testingSet[index] = list()
                    testingSet[index].append(x)
                elif count < halfset and index in testingSet:
                    testingSet[index].append(x)
                elif count >= halfset and index not in trainingSet:
                    trainingSet[index] = dict()
                    trainingSet[index][x] = y
                elif count >= halfset and index in trainingSet: 
                    trainingSet[index][x] = y
                    
            count = count + 1
                
                    
    for k, v in core_set_venues.items():
        index = 0
        for x, y in v.items():
            if x == 'index':
                index = y
            if x == 'num_comments':
                venues[index] = y
                if y > maxTips_Venue:
                    maxTips_Venue = y
                    
    print ("Max number of tips by a user: "+str(maxTips_User))
    print ("Max number of tips for a venue: "+str(maxTips_Venue))   
    print ("Number of users with single tip: "+str(len(trainingSet['Singles'])))
    
    comments_training = list()
    
    for k, v in trainingSet.items():
        if k == 'Singles':
            for x, y in v.items(): 
                for a, b in y.items():
                    comments_training.append(b)
                
        else :
            comments_training.extend(list(v.values()))
            
    
    #print (comments)
    
    #print (sumOfComments/numUsers)
    test_sum = 0
    test_sum = test_sum + len(trainingSet['Singles'])
    
    
    for k, v in trainingSet.items():
        if k != 'Singles':
            test_sum = test_sum + len(v)
    
    for k, v in testingSet.items():
        test_sum = test_sum + len(v)
    
    

    [dictionary, corpus] = createMainCorpus(comments_training)
    
     
    
    tfidf = getTfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    num_topics = 100
    
    #topWords = dictionary.filter_extremes(no_below = 20, no_above=0.9)
    
    lda = getLDAModel(corpus, dictionary, num_topics)
  
#    # nearest-neighbors
    k = 50
    
    num_recommended = 0
    num_matches = 0
    
    
    
    numbers = dict()
    numbers['num_topics'] = num_topics
    numbers['num_users'] = num_users
    numbers['num_venues'] = num_venues
    
    count = 0 
    
    relevant = 0
    
    a = [0.0] * num_topics
        
    lda_corpus = lda[corpus]

    for doc in lda_corpus:
        for id, score in doc:
            a[id] = a[id]+score;

    for i in range(len(a)):
        a[i] = a[i]/len(corpus)
        
    topic_id = a.index(max(a))
    
    P_matrix = getPMatrix(trainingSet, topic_id, dictionary, num_users, num_venues, lda, tfidf)
    print (P_matrix.nonzero())

    P_prime = computeSVD(P_matrix)
    #print (P_prime.nonzero())
    sim_matrix = np.corrcoef(P_prime)
    
    #sim_matrix = cosine_similarity(P_matrix)#np.corrcoef(P_matrix)#computeSimilarityMatrix(P_prime, num_users)
    print (sim_matrix.nonzero())
    
   
    sum_precision = 0
    sum_recall = 0
    
    for x in testingSet.keys():
        
        similar_users = np.array(sim_matrix[x])
        nearestNeighbors = kNearest(x, similar_users, k)
    
        recommended_venues = recommend(nearestNeighbors, trainingSet)
        
        print (recommended_venues)
        recommended_venues = removeDuplicates(trainingSet, recommended_venues, x)
        print (recommended_venues)
        [matches, relevant_temp] = evaluate(testingSet, recommended_venues, x)
#
    
        
        print (matches)
        print (relevant_temp)
        relevant = relevant + relevant_temp
        num_recommended = num_recommended + len(recommended_venues)
        num_matches = num_matches + matches
        
        precision = num_matches/num_recommended
        recall = num_matches/relevant
        
        sum_precision = sum_precision + precision
        sum_recall = sum_recall + recall
        
        count = count + 1
        print (count)
        
        
    precision = sum_precision/count
    recall = sum_recall/count
    
    print(num_users)
    print(num_venues)
    print ("Values")
    print ('K = '+str(k))
    print ('#Top= '+str(num_topics))
    print (precision)
    print (recall)
    print (num_matches)
    print (num_recommended)
    print (relevant)



    

main_alt()
#main()    
#test()   