# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:29:39 2017

@author: neelu
"""
import pandas as pd
import ujson as json
import csv

#df = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json')

#df.drop(df.head(1).index, inplace=True)
#df.to_csv('review.csv')


with open('yelp_dataset/yelp_academic_dataset_review.json', 'rb') as json_file:
    data = json_file.readlines()
    
data = map(lambda x: x.rstrip(), data)

data_json_str = "["+','.join(data)+"]"

data_df = pd.read_json(data_json_str)
