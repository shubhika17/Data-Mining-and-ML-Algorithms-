# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:49:36 2017

@author: Shubhika
"""

import pandas as pd
import random
import sys
import math
import numpy as np
import matplotlib.pyplot as grp 

def loadfile(filename):
        data = pd.read_csv(filename,names = ["ReviewId", "ClassLabel", "ReviewText"],delimiter="\t", encoding='utf-8')
        data['ReviewText'] = data['ReviewText'].str.lower()
        data['ReviewText'] = data['ReviewText'].str.replace('[^\w\s]','')
        document = {}
        for i in range(len(data)):
            x = data.loc[[i]]
            y = x['ReviewText'].str.split().tolist()
            y = y[0]
            y = set(y)
        #documentinfo = [x['ReviewId'], x['ClassLabel'], y]
            ID = x.get_value(index = i, col = 'ReviewId')
            ClassLabel = x.get_value(index = i, col = 'ClassLabel')
            document[ID] = [ClassLabel, y]
        return document

def mostfreqwords(document, numberofword):
    wordlist = {}
    for ID in document:
        for word in document[ID][1]: 
            if word not in wordlist:
                wordlist[word] = {}
                wordlist[word] = 1
            else:
                wordlist[word] += 1
    wordlist = list(reversed(sorted(wordlist.items(), key=lambda item: item[1])))
    if(len(wordlist) >= 4100):
        wordlist = wordlist[100: (numberofword + 100)]
    else:
        wordlist = wordlist[100: ]
    return wordlist

def featurevector(document, wordlist, val):
    for ID in document:
        #print(ID)
        vector = []
        reviewWords = []
        reviewWords = document[ID][1]
        #print(reviewWords)
        #print(len(wordlist))
        vector.append(1)
        for word in wordlist:
            if val == True:
                count = 0
                for i in range(len(reviewWords)):
                    fea = reviewWords[i]
                    if word[0] == fea:
                        count += 1
                if count > 1:
                   vector.append(2)
                elif count == 1:
                    vector.append(1)
                else:
                    vector.append(0)
            else:
                if word[0] in reviewWords:
                    vector.append(1)
                else:
                    vector.append(0)
        document[ID].append(vector)
    #print(document[ID])
def xValues(document):
     feature = document[0][2]
     x = np.zeros(shape=(len(document),len(feature)))
     j = 0
     for ID in document:
         for i in range(len(feature)):
             x[j][i] = document[ID][2][i]
         j = j + 1
     return x

def yValues(document):
     x = np.zeros(shape=(len(document)))
     j = 0
     for ID in document:
         if document[ID][0] == 0:
            x[j] = -1 
         else:
             x[j] = document[ID][0]
         j = j + 1
     return x
 
def val(weights,x):
    length = np.shape(x)
    #print(length)
    y_prime = np.zeros(shape=(length[0]))
    for i in range(length[0]):
        x_val = x[i, :]
        y_prime[i] = np.dot(weights, x_val)
    return y_prime

def svm_grad(weights, x, y):
    y_prime = val(weights, x)
    delta = np.zeros(shape=(len(weights)))
    for i in range(len(weights)):
        temp_delta = 0
        for j in range(len(y)):
            if(y[j]*y_prime[j] < 1):
                temp_delta += y[j]*x[j][i]
        temp_delta =  0.01*weights[i] - temp_delta
        temp_delta = (float(1)/(len(y)))*temp_delta
        delta[i] = temp_delta
    return delta
def gradient_descent(y, x, weights):
    changeInWeights = 1
    i = 1
    while((changeInWeights > 1e-6) & (i <= 100)):
        print(i)
        previous_weights = weights
        weights = weights - (0.5 * svm_grad(weights, x, y))
        changeInWeights = np.abs(np.linalg.norm(previous_weights) - np.linalg.norm(weights))
        i+=1
    return weights

def prediction_svm(theta, X, classLabel):
    pred_prob = val(theta, X)
    pred_value = np.where(pred_prob > 0, 1, -1)
    misclassification = np.sum(classLabel != pred_value)
    print("ZERO-ONE-LOSS ",misclassification/len(pred_value))    

train = loadfile('yelp_data.csv')
wordlist = mostfreqwords(train, 4000)
featurevector(train, wordlist,False)
X = xValues(train)
classLabel = yValues(train)
shape = X.shape[1]
coeff = np.zeros(shape)
coeff = gradient_descent(classLabel, X, coeff)
prediction_svm(coeff, X, classLabel)
    
        
    