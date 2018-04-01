# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:03:27 2017

@author: neelu
"""
import numpy as np
import csv 
import sys

# data in text file to csv file with 3 columns --> userID, venueID, comment

txt_file = r"dataset_ubicomp2013_tips.txt"
csv_file = r"training.csv"
#
f1 = open(txt_file, "r")
f2 = open(csv_file, "w")
in_txt = csv.reader( f1, delimiter='\t')
out_csv = csv.writer(f2)

out_csv.writerows(in_txt)

f1.close()
f2.close()

#users = list()
#venues = list()
#comments = list()
#
#with open("./training.csv", "r") as trainFile:
#    reader = csv.reader(trainFile)
#    
#    for row in reader:
#        if len(row) != 0:
#            users.append(int (row[0]))
#            venues.append(int (row[1]))
#            comments.append((row[2]))
            

#users = list()
#venues = list()
#comments = list()
#max_user = 0
#max_venue = 0
#with open("dataset_ubicomp2013_tips.txt") as f:
#    for line in f:
#        row = line.split()
#        temp0 = int (row[0])
#        temp1 = int (row[1])
#        if temp0 > max_user:
#            max_user = temp0
#        if temp1 > max_venue:
#            max_venue = temp1
#        users.append(int(row[0]))
#        venues.append(int(row[1]))
#        l = len(row)
#        comments.append(" ".join(row[2:l]))
#
#user = np.zeros(max_user+1)
#venue = np.zeros(max_venue+1)
#print(max_user)
#print(max_venue)
#
#for i in users:
#    
#    user[i] = user[i]+1
#
#for i in venues:
#    venue[i] = venue[i]+1
#
#print(venue)
#
#f = open("training.csv", 'wt')
#try:
#    writer = csv.writer(f)
#    writer.writerow( ("Users", "Venues", "Comments") )
#    for i in range(len(users)):
#        temp1 = users[i]
#        temp2 = venues[i]
#        if user[temp1] >= 30 and venue[temp2] >= 80:
#            writer.writerow( (temp1, temp2, comments[i]) )
#finally:
#    f.close()


        

    
    