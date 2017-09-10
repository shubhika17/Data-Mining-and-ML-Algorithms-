# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import random
import sys
import math
import numpy
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
    wordlist = wordlist[100: (numberofword + 100)]
    print("WORD1 ",wordlist[0][0])
    print("WORD2 ",wordlist[1][0])
    print("WORD3 ",wordlist[2][0])
    print("WORD4 ",wordlist[3][0])
    print("WORD5 ",wordlist[4][0])
    print("WORD6 ",wordlist[5][0])
    print("WORD7 ",wordlist[6][0])
    print("WORD8 ",wordlist[7][0])
    print("WORD9 ",wordlist[8][0])
    print("WORD10 ",wordlist[9][0])
    return wordlist

def featurevector(document, wordlist):
    for ID in document:
        #print(ID)
        vector = []
        reviewWords = []
        reviewWords = document[ID][1]
        #print(reviewWords)
        for word in wordlist:
            #print(word[0])
            if word[0] in reviewWords:
                vector.append(1)
            else:
                vector.append(0)
        document[ID].append(vector)
    #print(document[ID])

def probability(wordlist, document):
    total = len(document)
    masterlist = []
    list0_0 = []
    list1_0 = []
    list0_1 = []
    list1_1 = []    
    count = 0
    for word in wordlist :
        #print('xxxx: '+ str(len(wordlist)))
        #print('count: '+ str(count))
        count0_0 = 0.0
        count1_0 = 0.0
        count0_1 = 0.0
        count1_1 = 0.0
        for ID in document:
            feature = []
            feature = document[ID][2]
            #print('len: ' + str(len(feature)))
            if document[ID][0] == 1:
                #print(feature[count])
                #print(count)
                if feature[count] == 0:
                    #print(count)
                    count1_0 += 1
                else:
               #print('here')
                   count1_1 += 1
            else:
                if feature[count] == 0:
                    count0_0 += 1
                else:
                #print('here')
                    count0_1 += 1            
        prob0_0 = (count0_0 + 1)/(count0_0 + count0_1 + 2)
        prob1_0 = (count1_0 + 1)/(count1_0 + count1_1 + 2)
        prob0_1 = (count0_1 + 1)/(count0_0 + count0_1 + 2)
        prob1_1 = (count1_1 + 1)/(count1_0 + count1_1 + 2)
        list0_0.append(prob0_0)
        list1_0.append(prob1_0)
        list0_1.append(prob0_1)
        list1_1.append(prob1_1)
        count += 1
    prob0 = (count0_0 + count0_1)/total
    prob1 = (count1_0 + count1_1)/total
    masterlist.append(list0_0)
    masterlist.append(list0_1)
    masterlist.append(list1_0)
    masterlist.append(list1_1)
    masterlist.append(prob0)
    masterlist.append(prob1)
    return masterlist

def prediction(document, masterlist):
    misclassify=0.0
    basemisclassify = 0
    total=len(document)
    for ID in document:
        probC0=1.0
        probC1=1.0
        classLabel=document[ID][0]
        baseLabel = 0
        for i in range(len(document[ID][2])):
            #print('doc len: '+ str(len(document[ID][2])))
            #print('masterlist: ' + str(len(masterlist[0])))
            if document[ID][2][i] == 0:
                probC0=probC0*masterlist[0][i]
                probC1=probC1*masterlist[2][i]
            else:
                probC0=probC0*masterlist[1][i]
                probC1=probC1*masterlist[3][i]
        probC0=probC0*masterlist[4]
        probC1=probC1*masterlist[5]
        if(probC0 > probC1):
            predClassLabel = 0
        else:
            predClassLabel = 1
        if(classLabel!=predClassLabel):
            misclassify=misclassify+1
        
        if masterlist[4] > masterlist[5]:
            baseLabel = 0
        else:
            baseLabel = 1
        if(classLabel!=baseLabel):
            basemisclassify = basemisclassify +1
        
    mislclassification=misclassify/total
    basemislclassification = basemisclassify/total
    print("ZERO-ONE-LOSS ",mislclassification)
    pred = []
    pred.append(mislclassification)
    pred.append(basemislclassification)
    return pred

def datapartition(document, percent):
    size=math.ceil(percent*0.01*len(document))
    TrainingData={}
    TestData={}
    keys=list(document.keys())
    random.shuffle(keys)
    for k in range(size):
        TrainingData[keys[k]]=document[keys[k]]
    for k in range(size+1,len(document)):
        TestData[keys[k]]=document[keys[k]]
    data = []
    data.append(TrainingData)
    data.append(TestData)
    return data

def question3(document):
    percentage =  [1, 5, 10, 20, 50, 90]
    average = []
    standdev = []
    baseLoss = []
    basestd = []
    for percent in percentage:
        zeroOneLoss = []
        baseLosses = []
        for i in range(10):
            data = datapartition(document, percent)
            trainingdata = data[0]
            testdata = data[1]
            wordlist = mostfreqwords(trainingdata,500)
            featurevector(trainingdata, wordlist)
            featurevector(testdata, wordlist)
            masterlist = probability(wordlist, trainingdata) 
            pred = prediction(testdata, masterlist)
            zeroOneLoss.append(pred[0])
            baseLosses.append(pred[1])
        average.append(numpy.average(zeroOneLoss))
        standdev.append(numpy.std(zeroOneLoss))
        baseLoss.append(numpy.average(baseLosses))
        basestd.append(numpy.std(baseLosses))
    print(str(len(baseLoss)))
    print(str(len(percentage)))
    grp.figure(1)
    grp.errorbar(percentage, average, standdev,  marker='^',  label = "NBC 0-1 loss")
    grp.errorbar(percentage, baseLoss, basestd,  marker='^',  label = "baseline 0-1 loss")
    grp.xlabel('Percentage of Training Data')
    grp.ylabel('Zero-One Loss')
    grp.legend()
    grp.savefig('training_size_loss_q3.png')

def question4(document):
    features =   [10, 50, 250, 500, 1000, 4000]
    average = []
    standdev = []
    baseLoss = []
    basestd = []
    for feature in features:
        zeroOneLoss = []
        baseLosses = []
        for i in range(10):
            data = datapartition(document, 50)
            trainingdata = data[0]
            testdata = data[1]
            wordlist = {}
            wordlist = mostfreqwords(trainingdata,feature)
            featurevector(trainingdata, wordlist)
            featurevector(testdata, wordlist)
            masterlist = []
            masterlist = probability(wordlist, trainingdata) 
            pred = []
            pred = prediction(testdata, masterlist)
            zeroOneLoss.append(pred[0])
            baseLosses.append(pred[1])
        average.append(numpy.average(zeroOneLoss))
        standdev.append(numpy.std(zeroOneLoss))
        baseLoss.append(numpy.average(baseLosses))
        basestd.append(numpy.std(baseLosses))
    grp.figure(1)
    grp.errorbar(features, average, standdev,  marker='^',  label = "NBC 0-1 loss")
    grp.errorbar(features, baseLoss, basestd,  marker='^',  label = "baseline 0-1 loss")
    grp.xlabel('Number of Features')
    grp.ylabel('Zero-One Loss')
    grp.legend()
    grp.savefig('features_loss_q4.png')
    
    
q3_4 = False

if (len(sys.argv) == 3):
    trainingData = sys.argv[1]
    testData = sys.argv[2]
    train = loadfile(trainingData)
    test = loadfile(testData)
    wordlist = mostfreqwords(train, 4000)   
    featurevector(train, wordlist)
    featurevector(test, wordlist)
    masterlist = probability(wordlist, train) 
    prediction(test, masterlist)
elif q3_4 == True:
    trainingData = sys.argv[1]
    train = loadfile(trainingData)
    question3(train)
    question4(train)
    
else:
    print("3 arguments needed. Invalid input!!")
    exit()


