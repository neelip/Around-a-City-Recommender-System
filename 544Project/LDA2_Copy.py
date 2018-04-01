# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:20:10 2017

@author: neelima potharaj 

NOTE: "LDA2.py" Copy - 5/1/2017
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
import pylab as pl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle

def getMatchCount(Pmain_vector, P2_vector):
    count_relevant = 0
    for i in range(0, len(Pmain_vector)):
        if Pmain_vector[i] != "" and P2_vector[i] != "":
            count_relevant = count_relevant + 1
        elif P2_vector[i] != "":
            count_retrieved = count_retrieved + 1
    count_retrieved = count_retrieved + count_relevant
    return [count_relevant, count_retrieved]

#def recommend(similarity_sparse, P, index, k):
    
    # get the similarity vector for the index
    #similarity_main = np.array(similarity_sparse[index].toarray())
    
    # get the top k users in a vector
    #k_indices = [-1]*k
    #max_val_index = -1
    #for i in range(0, k):
        
        #for i in range(0, len(similarity_main)):
            
                
    # count precision for given k users
        
    
    # 


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

def prepareUser(index, core_set):
    user_key = random.choice(list(core_set.keys()))
    user_comments = list()
    for k,v in core_set[user_key].items():
        if isinstance(k, int):
            user_comments.append(v)
        
    return user_comments

def createMainCorpus(comments):
    stopWords = set(stopwords.words('english'))
    words = [[word.lower() for word in comment.lower().split() if word.lower() not in stopWords and word.isalpha()] for comment in comments ]
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

def readFile_Original():
    users = list()
    venues = list()
    comments = list()
    
    core_set_users = {}
    core_set_venues = {}
    
    count_users = 0
    count_venues = 0


    with open("./training.csv", "r") as trainFile:
        reader = csv.reader(trainFile)
        
        for row in reader:
            if len(row) != 0:
                
                if row[0] not in core_set_users :
                    
                    core_set_users[row[0]] = {}
                    core_set_users[row[0]]["index"] = count_users
                    core_set_users[row[0]]["num_comments"]  = 0
                    count_users = count_users + 1
                    if row[1] not in core_set_venues :
                        core_set_venues[row[1]] = {}
                        core_set_venues[row[1]]["index"] = count_venues
                        core_set_venues[row[1]]["num_comments"] = 0
                        count_venues = count_venues + 1
                if row[1] not in core_set_venues :
                    count_venues = count_venues + 1
                    core_set_venues[row[1]] = {}
                    core_set_venues[row[1]]["index"] = count_venues
                    core_set_venues[row[1]]["num_comments"] = 0
#==============================================================================
#                 print("users ",count_users)
#                 print("venues",count_venues)
#==============================================================================
                

                if core_set_venues[row[1]]["index"] in core_set_users[row[0]]:
                    core_set_users[row[0]][core_set_venues[row[1]]["index"]] = core_set_users[row[0]][core_set_venues[row[1]]["index"]] + row[2]
                    
                elif core_set_venues[row[1]]["index"] not in core_set_users[row[0]]:
                    core_set_users[row[0]][core_set_venues[row[1]]["index"]] = row[2]
                    core_set_users[row[0]]["num_comments"] = core_set_users[row[0]]["num_comments"] + 1
                if core_set_users[row[0]]["index"] in core_set_venues[row[1]]:
                    core_set_venues[row[1]][core_set_users[row[0]]["index"]] = core_set_venues[row[1]][core_set_users[row[0]]["index"]] + row[2]
                    
                elif core_set_users[row[0]]["index"] not in core_set_venues[row[1]]:
                    core_set_venues[row[1]][core_set_users[row[0]]["index"]] = row[2]
                    core_set_venues[row[1]]["num_comments"] = core_set_venues[row[1]]["num_comments"] + 1

        
                comments.append((row[2]))
            
    #users = users[0:10000]
    #venues = venues[0:10000]
    #comments = comments[0:10000]
#    
    output = open('core_set_users.txt', 'ab+')
    pickle.dump(core_set_users, output)
    output.close()
#    
    output2 = open('core_set_venues.txt', 'ab+')
    pickle.dump(core_set_venues, output2)
    output2.close()
#    
    output3 = open('comments.txt', 'ab+')
    pickle.dump(comments, output3)
    output3.close()
#    
    return [core_set_users, core_set_venues, comments]


def main():
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    [core_set_users, core_set_venues, comments] = readFile_Original()
    
#    init_matrix = np.chararray((len(core_set_users)+1, len(core_set_venues)+1))
##    
##    
#    init_matrix = np.chararray(init_matrix.shape, itemsize=10)
##    
##    
#    users = [0 for x in range(len(core_set_users)+1)]
#    venues = [0 for x in range(len(core_set_venues)+1)]
#    
#    maxTips_User = 0
#    maxTips_Venue = 0
##    
##    
##    
#    for k, v in core_set_users.items():
#        index = 0
#        for x, y in v.items():
#            if x == "index":
#                index = y
#            if x == "num_comments":
#                users[index] = y
#                if y > maxTips_User: 
#                    maxTips_User = y
#            if isinstance(x, int):
#                
#                init_matrix[index, x] = y.encode('ascii', 'ignore')
#    
#    for k, v in core_set_venues.items():
#        index = 0
#        for x, y in v.items():
#            if x == "index":
#                index = y
#            if x == "num_comments":
#                venues[index] = y
#                if y > maxTips_Venue:
#                    maxTips_Venue = y
##                
#    print (maxTips_User)
#    print (maxTips_Venue)
##    # create and save distribution graphs - premodification                  
##    #graph(users, venues, False)
##    
##
##    
##    #for i in range(len(core_set_users)+1):
##        
##    
##       # print (len(np.nonzero(init_matrix[i])[0]))
##              
##        #print (users[i])
##        #print (venues[i])
##        
##     # Modification
###    for i in range(100+1):
###        if len(np.nonzero(init_matrix[i])[0]) <= 10:
###            index = np.nonzero(init_matrix[i])[0]
###            print (i)
###            for j in index:
###                if len(np.nonzero(init_matrix[:,j])[0]) <= 10:
###                    np.delete(init_matrix, [j], axis = 1)
###                    np.delete(init_matrix, [i], axis = 0)
###                    users[i] = 0
###                    venues[j] = 0
###                      
###    print (users[0:100])
###    print (venues[0:100])
###    graph(users[0:100], venues[0:100], True)
###    
##
##    
##
#    [dictionary, corpus] = createMainCorpus(comments)
#    index = 1782
#     
#    user_comments = prepareUser(index, core_set_users)
#    user_corpus = [dictionary.doc2bow(comment.lower().split()) for comment in user_comments]
#     
#    tfidf = getTfidfModel(corpus)
#    corpus_tfidf = tfidf[corpus]
##     
#    num_topics = 10
#     
#    lda = getLDAModel(corpus_tfidf, dictionary, num_topics)
#     
#    topic_id = getTopic(lda, user_corpus, num_topics)
##     
#    P_Matrix = sp.dok_matrix((len(core_set_users)+1, len(core_set_venues)+1), dtype=np.float32)
#     
#    for i, j in core_set_users.items():
#        x = 0
#        for k, v in j.items():
#            
#            
#            if k == "index":
#                x = v
#                
#            if isinstance(k, int):
#                temp_comment = v
#                temp_corpus = dictionary.doc2bow(temp_comment.lower().split())
#                temp_lda = lda[temp_corpus]
#                temp_theta = 0
#             
#                for doc1, doc2 in temp_lda:
#                    if doc1 == topic_id:
#                        temp_theta = float (doc2)
#                        print (temp_theta)
#                 
#                P_Matrix[x, k] = temp_theta   
#                      
#  
#    P = csc_matrix(P_Matrix, dtype=np.float32)
##     
#    [U, s, Vt] = sparsesvd(P, 100)
#    dim = (len(s), len(s))
#    S = np.zeros(dim, dtype = np.float32)
#    for i in range(0, len(s)):
#        S[i, i] = mt.sqrt(s[i])
#    U = csc_matrix(np.transpose(U), dtype=np.float32)
#    S = csc_matrix(S, dtype=np.float32)
#    Vt = csc_matrix(Vt, dtype = np.float32)
#     
#    print (U.shape)
#    print (S.shape)
#    print (Vt.shape)
#     
#    P_prime = U*S*Vt
#     
#    print (P_prime.shape)
#    
#    
#    PP_sparse = sp.csr_matrix(P_prime)
##    similarity_sparse = cosine_similarity(PP_sparse, dense_output=False)
##    similarity_user = similarity_sparse[1782,:].toarray()
##    recommend(similarity_sparse, P, index)
##    #print ('pairwise dense output:\n {}\n'.format(similarity_sparse))
##    #print (similarity_sparse.shape)
    

def main_alt():
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    [core_set_users, core_set_venues, comments] = readFile()
    
    # Create data structures for modification and graphing 
    
    init_matrix = np.chararray((len(core_set_users)+1, len(core_set_venues)+1))
    
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
    
    dictionary = 
    
    num_topics = 100
    
    lda = getLDAModel(corpus_tfidf, dictionary, num_topics)
  
#    # nearest-neighbors
    k = 5
    
    num_recommended = 0
    num_matches = 0
    
    
    
    numbers = dict()
    numbers['num_topics'] = num_topics
    numbers['num_users'] = num_users
    numbers['num_venues'] = num_venues
    
    count = 0 
    
    relevant = 0
    
    for x in testingSet.keys():
        
        
        
        recommended_venues = getRecommendations(x, lda, trainingSet, k, dictionary, numbers)
        print (recommended_venues)
        recommended_venues = removeDuplicates(trainingSet, recommended_venues, x)
        print (recommended_venues)
        [matches, relevant] = evaluate(testingSet, recommended_venues, x)
#        
        
        print (matches)
        print (relevant)
        num_recommended = num_recommended + len(recommended_venues)
        num_matches = num_matches + matches
        
        
        count = count + 1
        print (count)
        if count == 5:
            break
        
    #precision = num_matches/num_recommended
    #recall = num_matches/relevant
    
    #print (precision)
    #print (recall)

def test():
    mat = sp.rand(200, 100, density=0.01) # create a random matrix
    smat = csc_matrix(mat) # convert to sparse CSC format
    ut, s, vt = sparsesvd(smat, 1) # do SVD, asking for 100 factors
    mat_prime = np.dot(ut.T, np.dot(np.diag(s), vt))
    
    print (len(np.transpose(mat.nonzero())))
    print (len(np.transpose(mat_prime.nonzero())))

 
    
    
    
    
main()   