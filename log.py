# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:49:35 2017

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

def featurevector_2(document, wordlist, val):
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
        if len(document[ID]) < 3:           
            document[ID].append(vector)
        else:
            document[ID][2] = vector            
    #print(document[ID])
def xValues(document):
     for Id in document:
         feature = document[Id][2]
         break
     x = np.zeros(shape=(len(document),len(feature)))
     j = 0
     for ID in document:
         for i in range(len(feature)):
             x[j][i] = document[ID][2][i]
         j = j + 1
     return x

def yValues_log(document):
     x = np.zeros(shape=(len(document)))
     j = 0
     for ID in document:
         x[j] = document[ID][0]
         j = j + 1
     return x

def log_eq(weights, x):
    length = np.shape(x)
    #print(length)
    y_prime = np.zeros(shape=(length[0]))
    for i in range(length[0]):
        x_val = x[i, :]
        y_prime[i] = float(1) / (1 + math.exp(-(np.dot(x_val, weights))))
    return y_prime

def log_grad(y, x, weights):
    #print(np.shape(weights))
    y_prime = log_eq(weights, x)
    delta = np.zeros(shape=(len(weights)))
    for i in range(len(weights)):
        temp_delta = 0
        for j in range(len(y)):
             temp_delta += (y[j] - y_prime[j])*x[j][i]   
        temp_delta = temp_delta - 0.01*weights[i]
        delta[i] = temp_delta
    return delta

def gradient_descent(y, x, weights):
    changeInWeights = 1
    i = 1
    while((changeInWeights > 1e-6) & (i <= 100)):
        print(i)
        previous_weights = weights
        weights = weights + (0.01 * log_grad(y, x, weights))
        changeInWeights = np.abs(np.linalg.norm(previous_weights) - np.linalg.norm(weights))
        i+=1
    return weights         
def prediction_log(theta, X, classLabel):
    pred_prob = log_eq(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    misclassification = np.sum(classLabel != pred_value)
    print("ZERO-ONE-LOSS ",misclassification/len(pred_value)) 
    
def datapartition(document):
    percent =  [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    data = {}
    keys=list(document.keys())
    random.shuffle(keys)
    count = 0
    for k in range(10):
        data[k] = {}
        for i in range(200):
            data[k][keys[count]]=document[keys[count]]
            count += 1
    
    for j in percent:
            size=math.ceil(j*2000)
            for l in range(10):
                fulldata = {}
                count = 0
                for x in range(10):
                    if(x != l):
                        values = list(data[x].keys())
                        for i in range(200):
                            fulldata[values[i]] = {}
                            fulldata[values[i]] = [data[x][values[i]][0],data[x][values[i]][1]] 
                temp1=list(fulldata.keys())
                random.shuffle(temp1)
                train = {}
                test = {}
                #print(train)
                for i in range(size):
                    train[temp1[i]] = {}
                    train[temp1[i]] = [fulldata[temp1[i]][0],fulldata[temp1[i]][1]] 
                wordlist = mostfreqwords(train, 4000)
                temp =list(data[l].keys())
                for i in range(200):
                    test[temp[i]] = {}
                    test[temp[i]] = [data[l][temp[i]][0],data[l][temp[i]][1]]
                if(True):
                    featurevector_2(train, wordlist,False)
                    featurevector_2(test, wordlist, False)
                    X = xValues(train)
                    classLabel = yValues_log(train)
                    shape = X.shape[1]
                    coeff = np.zeros(shape)
                    coeff = gradient_descent(classLabel, X, coeff)
                    X_test = xValues(test)
                    classLabel_test = yValues_log(test)
                    prediction_log(coeff, X_test, classLabel_test)    

train = loadfile('yelp_data.csv')
datapartition(train)
wordlist = mostfreqwords(train, 4000)
featurevector_2(train, wordlist,False)
X = xValues(train)
classLabel = yValues_log(train)
shape = X.shape[1]
coeff = np.zeros(shape)
coeff = gradient_descent(classLabel, X, coeff)
print(coeff)
prediction_log(coeff, X, classLabel)
      

