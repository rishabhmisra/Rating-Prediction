# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:11:03 2017

@author: rmisra
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

user_item_count = {};   ## for each user, number of items recorded
item_user_count = {};   ## for each item, number of users recorded
target = [];    ## labels for train data
user_max=0;     ## maximum id for the user
item_max=0;     ## maximum id for the item

##--------------------------------read files----------------------------------------------

file = open('rating_train.txt','r')

## find total number of users and items and build up the dictionaries
for l in file.readlines():
	line = l.strip().split(',')
	user_id = int(line[0])
	item_id = int(line[1])
	rating = float(line[2])
	
	target.append(rating)
	
	if(user_max<user_id):
		user_max = user_id;
			
	if(item_max<item_id):
		item_max = item_id;
			
	if user_id not in user_item_count:
		user_item_count[user_id] = 1;
	else:
		user_item_count[user_id]+=1;
			
	if item_id not in item_user_count:
		item_user_count[item_id] = 1;
	else:
		item_user_count[item_id]+=1;
		
file.close()

test_target = []    ## labels for test data
file = open('rating_val.txt','r')

for l in file.readlines():
	line = l.strip().split(',')
	user_id = int(line[0])
	item_id = int(line[1])
	rating = float(line[2])
	test_target.append(rating)
	
	if(user_max<user_id):
		user_max = user_id;
			
	if(item_max<item_id):
		item_max = item_id;

file.close()

num_rows = len(target);
num_users = user_max+1;
num_items = item_max+1;
print('number of users: ' + str(num_users) + ' number of items: ' + str(num_items))

##--------------------------------------------------------

# Create the rating matrix
R = np.zeros((num_users, num_items))

file = open('rating_train.txt','r')
for l in file.readlines():
    line = l.strip().split(',')
    R[int(line[0]), int(line[1])] = float(line[2])

val_label = []
for l in open("rating_val.txt"):
    u,i,r = l.strip().split(',')
    val_label.append(float(r))
    

mu = 4.23359    # global bias - average of all the train data labels
lmbda = 1 # Regularisation weight
k = 5  # Dimension of the latent feature space
m, n = R.shape  # Number of users and items
n_epochs = 100  # Number of epochs
gamma = 0.01  # Learning rate
mul = 0.1   # multiplication factor

    
beta_u = mul*np.zeros((num_users,1))    ## user bias
beta_i = mul*np.zeros((num_items,1))    ## item bias


P =  mul*np.random.randn(k,m) # Latent user feature matrix
Q =  mul*np.random.randn(k,n) # Latent movie feature matrix

beta_u = beta_u.flatten()
beta_i = beta_i.flatten()

#Only consider non-zero matrix 
users,items = R.nonzero()     
save=1000000000 

## ----------------------------------optimize---------------------------------------------
for epoch in range(n_epochs):
    for u, i in zip(users,items):

        e = R[u, i] - (mu + beta_u[u] + beta_i[i] + np.dot(P[:,u].T,Q[:,i]))  # Calculate error for gradient

        beta_i[i] += gamma*(e - lmbda*beta_i[i])
        beta_u[u] += gamma*(e - lmbda*beta_u[u])
        P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
        Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix


    ## test performance on validation data
    index = 0
    diff = 0
    for l in open("rating_val.txt"):
        u,i,r = l.strip().split(',')
        u = int(u)
        i = int(i)    
        rating = min(5,(mu+ beta_u[u] + beta_i[i] + np.dot(P[:,u].T,Q[:,i])))
        diff += (val_label[index] - rating)**2
        index+=1

    mse = diff/len(val_label)
    print(epoch, mse)
    
    if(mse > save and epoch>=10):
        break
    
    save = mse