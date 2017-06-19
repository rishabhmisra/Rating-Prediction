# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:21:31 2017

@author: rmisra
"""
import gzip

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

user = {}
item = {}
user_index = 0
item_index = 0

##----------------------------convert user and item id in sequence (in form of 1 to n)
for l in readGz("train.json.gz"):
    
    if l['reviewerID'] not in user:
        user[l['reviewerID']] = user_index; user_index+=1
    
    if l['itemID'] not in item:
        item[l['itemID']] = item_index; item_index+=1
        

## test file
file = open('pairs_Rating.txt', 'r')

for l in file.readlines():
    if l.startswith("userID"):
        continue

    u,i = l.strip().split('-')
    if u not in user:
        user[u] = user_index; user_index+=1
    
    if i not in item:
        item[i] = item_index; item_index+=1
file.close()


##---------------- preparing training data file for rating prediction---------------------
file = open("rating_train_whole.txt", 'w')

for l in readGz("train.json.gz"):
    file.write(str(user[l['reviewerID']]) + ',' + str(item[l['itemID']]) + ',' + str(l['rating']) + '\n')


## test data for rating prediction
file1 = open("rating_test.txt", 'w')

for l in open('pairs_Rating.txt', 'r'):
    if l.startswith("userID"):
        continue
    u,i = l.strip().split('-')
    file1.write(str(user[u]) + ',' + str(item[i])+ '\n')

file.close()
file1.close()


##---------------- splitting given data into training and validation for model training-------------------
import random
seed = 3
random.seed(seed)
shuffle = [i for i in range(0,60000)]
random.shuffle(shuffle)

training_size = 180000
validation_size = 20000
        
file = open('rating_train.txt','w')
file1 = open('rating_val.txt','w')


counter = -1
for l in readGz("train.json.gz"):
    counter+=1
    if (shuffle[counter]<training_size):
        file.write(str(user[l['reviewerID']]) + ',' +  str(item[l['itemID']]) + ',' + str(l['rating']) + '\n')
    else:
        file1.write(str(user[l['reviewerID']]) + ',' +  str(item[l['itemID']]) + ',' + str(l['rating']) + '\n')
            
        
file.close()
file1.close()