# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:38:01 2017

@author: Shubhika
"""
import pandas as pd
import random
import sys
import math
import numpy as np
import matplotlib.pyplot as grp 
#from sklearn import tree

def loadfile(filename):
        data = pd.read_csv(filename,names = ["ReviewId", "ClassLabel", "ReviewText"],delimiter="\t", encoding='utf-8')
        data['ReviewText'] = data['ReviewText'].str.lower()
        data['ReviewText'] = data['ReviewText'].str.replace('[^\w\s]','')
        document = {}
        for i in range(len(data)):
            x = data.loc[[i]]
            y = x['ReviewText'].str.split().tolist()
            y = y[0]
        #documentinfo = [x['ReviewId'], x['ClassLabel'], y]
            ID = x.get_value(index = i, col = 'ReviewId')
            ClassLabel = x.get_value(index = i, col = 'ClassLabel')
            document[ID] = [ClassLabel, y]
        return document

def mostfreqwords(document, numberofword):
    wordlist = {}
    for ID in document:
        y = document[ID][1]
        y = set(y)
        for word in y: 
            if word not in wordlist:
                wordlist[word] = {}
                wordlist[word] = 1
            else:
                wordlist[word] += 1
    wordlist = list(reversed(sorted(wordlist.items(), key=lambda item: item[1])))
    if(len(wordlist) >= 1100):
        wordlist = wordlist[100: (numberofword + 100)]
    else:
        wordlist = wordlist[100:]
    return wordlist

def featurevector(document, wordlist, val):
    for ID in document:
        #print(ID)
        vector = []
        reviewWords = []
        reviewWords = document[ID][1]
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
  
def table(document):
     for Id in document:
         feature = document[Id][2]
         break
     x = np.zeros(shape=(len(document),len(feature) + 1))
     j = 0
     for ID in document:
         for i in range(len(feature)):
             x[j][i] = document[ID][2][i]
         x[j][i+1] = document[ID][0]
         j = j + 1    
     return x

def table_boosting(document):
    for Id in document:
        feature = document[Id][2]
        break
    x = np.zeros(shape=(len(document),len(feature) + 2))
    j = 0
    for ID in document:
        for i in range(len(feature)):
            x[j][i] = document[ID][2][i]
        if document[ID][0] == 0:
            x[j][i+1] = -1
        else:
            x[j][i+1] = document[ID][0]
        x[j][i+2] = 0
        j = j + 1    
    return x

def data_split(index,dataset):
    left, right = list(), list()
    for row in dataset:
        #print(row)  
        if row[index] == 1:
            left.append(row)
        else:
            right.append(row)
    return left, right  

def gini_gain(column, classLabel, weight, bst):
    gain1x1=0.0
    gain0x1=0.0
    gain0x0=0.0
    gain1x0=0.0
    if bst == True:
        for i in range(len(classLabel)):
            if classLabel[i] == 1:
                if column[i] == 1:
                    gain1x1 += np.asscalar(weight[i])
                elif column[i] == 0: 
                    gain1x0 += np.asscalar(weight[i])
            elif classLabel[i] == -1:
                if column[i] == 1:
                    gain0x1 += np.asscalar(weight[i])
                elif column[i] == 0:
                    gain0x0 += np.asscalar(weight[i])
        
        
        gain_data = 1 - (math.pow(((gain0x0 + gain0x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0)),2) + math.pow(((gain1x0 + gain1x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0)),2))
        if (gain0x1 + gain1x1) == 0:
            gain_positive = 1
        else:
            gain_positive = 1 - (math.pow((gain0x1/(gain0x1 + gain1x1)),2) + math.pow(gain1x1/(gain0x1 + gain1x1),2))
        if (gain1x0 + gain0x0) == 0:
            gain_negative = 0
        else:
            gain_negative = 1 - (math.pow((gain0x0/(gain1x0 + gain0x0)),2) + math.pow(gain1x0/(gain0x0 + gain1x0),2))
        gain = gain_data - ((((gain1x1 + gain0x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0))*gain_positive) + (((gain0x0 + gain1x0)/(gain0x0 + gain0x1 + gain1x1 + gain1x0))*gain_negative))
        #if gain < 0:
            #print("-ve", (gain0x0 + gain0x1))
            #print("+ve", (gain1x0 + gain1x1))
            #print(gain)
        return gain
        
        
    
    for i in range(len(classLabel)):
        if classLabel[i] == 1:
            if column[i] == 1:
                gain1x1 += 1.0
            elif column[i] == 0: 
                gain1x0 += 1.0
        elif classLabel[i] == 0:
            if column[i] == 1:
                gain0x1 += 1.0
            elif column[i] == 0:
                gain0x0 += 1.0
    gain_data = 1 - (math.pow(((gain0x0 + gain0x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0)),2) + math.pow(((gain1x0 + gain1x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0)),2))
    gain_positive = 1 - (math.pow((gain0x1/max((gain0x1 + gain1x1),1)),2) + math.pow((gain1x1/max((gain0x1 + gain1x1),1)),2))
    gain_negative = 1 - (math.pow((gain0x0/max((gain1x0 + gain0x0),1)),2) + math.pow((gain1x0/max((gain0x0 + gain1x0),1)),2))
    gain = gain_data - ((((gain1x1 + gain0x1)/(gain0x0 + gain0x1 + gain1x1 + gain1x0))*gain_positive) + (((gain0x0 + gain1x0)/(gain0x0 + gain0x1 + gain1x1 + gain1x0))*gain_negative))
        
    #if gain < 0 :
        #print("-ve", (gain0x0 + gain0x1))
        #print("+ve", (gain1x0 + gain1x1))
        #print(gain)
        #print(gain_data)
        #print(gain0x0)
        #print(gain0x1)
        #print(math.pow((gain0x0/max((gain0x0 + gain0x1),1)),2))
        #print(math.pow((gain0x1/max((gain0x0 + gain0x1),1)),2))
        #print(gain_positive)
        #print(gain_negative)
    return gain 

def feature_selection(data, rf, bst, visited_nodes):
    maximum = 0
    #print(maximum)
    shape = np.array(data).shape
    index = -1
    columns = shape[1]
    if bst == True:
        for i in range(columns - 3):
            if i not in visited_nodes:
                gini = gini_gain(np.array(data)[:,i], np.array(data)[:, columns - 2], np.array(data)[:, columns - 1] , bst)
                #print(i,": ",gini)
                if gini >= maximum:
                    maximum = gini
                    index = i
        if index == -1:
            outcomes = leaf(data, bst)
            print("here")
            return outcomes
        dataset = data_split(index, data)
        visited_nodes.append(index)
        return {'index':index, 'value': 1, 'data': dataset, 'nodes_visit': visited_nodes}
        
    #print(shape)
    if rf == True:
        features = []
        fea_len = math.sqrt(columns - 1)
        #print("heerr",fea_len)
        while len(features) < fea_len:
            index = random.randrange(0,(columns - 2))
            if index not in features:
                features.append(index)
        for i in features:
            if i not in visited_nodes:
                gini = gini_gain(np.array(data)[:,i], np.array(data)[:, columns - 1], None, bst)
                #print(i,": ",gini)
                if gini >= maximum:
                    maximum = gini
                    index = i
        if index == -1:
            outcomes = leaf(data, bst)
            print("here")
            return outcomes
        dataset = data_split(index, data)
        visited_nodes.append(index)
        return {'index':index, 'value': 1, 'data': dataset,'nodes_visit': visited_nodes}
    
    for i in range(columns - 2):
        if i not in visited_nodes:
            gini = gini_gain(np.array(data)[:,i], np.array(data)[:, columns - 1], None, bst)
            #print(i,": ",gini)
            if gini >= maximum:
                maximum = gini
                index = i
    if index == -1:
            outcomes = leaf(data, bst)
            print("here")
            return outcomes
        
    dataset = data_split(index, data)
    visited_nodes.append(index)
    return {'index':index, 'value': 1, 'data': dataset, 'nodes_visit': visited_nodes}

def leaf(data, bst):
    if bst == True:
        predictions = [row[-2] for row in data]
        weights = [row[-1] for row in data]
        count1ve = 0
        count1 = 0
        prediction = 0
        i = 0
        for pred in predictions:
            if pred == 1:
                count1 += weights[i]
            elif pred == -1:
                count1ve += weights[i]
            i += 1
        if count1 >= count1ve:
            prediction = 1
        else:
            prediction = -1
        return prediction
    outcomes = [row[-1] for row in data]
    return max(set(outcomes), key=outcomes.count)
 

def split(node, depth, max_depth, rf, bst):
    left, right = node['data']
    visited_nodes = node['nodes_visit']
    del(node['nodes_visit']) 
    del(node['data'])
    if not left or not right:
        node['left'] = node['right'] = leaf((left + right), bst)
        return
    if depth >= max_depth:
        node['left'], node['right'] = leaf(left, bst), leaf(right, bst)
        return
    if len(right) <= 10:
        node['right'] = leaf(right, bst)
    else:
        node['right'] = feature_selection(right, rf,  bst,visited_nodes)
        #print(node['right'])
        split(node['right'], depth+1,max_depth, rf, bst)
        
    if len(left) <= 10:   
        node['left'] = leaf(left, bst)
    else:
        node['left'] = feature_selection(left, rf, bst,visited_nodes)
        split(node['left'], depth+1,max_depth, rf,  bst)

    
def build_tree(train,max_depth, rf,bst):
    visited_nodes = []
    root = feature_selection(train, rf,  bst, visited_nodes)
    split(root, 1,max_depth, rf,  bst)
    return root

def predict(node, row):
    if row[node['index']] == node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            #print("here")
            return predict(node['right'], row)
        else:
            #print(node['right'])
            return node['right']

def decisionTree(train, test, max_depth, rf, bst):
    root = build_tree(train,max_depth, rf,  bst)
    test_shape = test.shape
    classLabel = test[:,test_shape[1] - 1]
    count = 0
    accuracy = 0
    #print(root)
    for row in test:
        prediction = predict(root, row)
        #print(prediction)
        if prediction == classLabel[count]:
            accuracy += 1
        count += 1
    zeroOneLoss = 1 - (accuracy/count)
    print("ZERO-ONE-LOSS-DT: ", zeroOneLoss)
    return zeroOneLoss


def pseudosample(data):
    sample = []
    while len(sample) < len(data):
        sample.append(random.choice(data))
    return sample


def bagging(train, test, max_depth, num_of_trees, rf, bst):
    test_shape = test.shape
    classLabel = test[:,test_shape[1] - 1]
    roots = []
    accuracy = 0
    count = 0
    for i in range(num_of_trees):
        sample = pseudosample(train)
        root = build_tree(sample, max_depth, rf, bst)
        roots.append(root)
    for row in test:
        predictions = [predict(root, row) for root in roots]
        count0 = 0
        count1 = 0
        for pred in predictions:
            if pred == 1:
                count1 += 1
            elif pred == 0:
                count0 += 1 
        if count1 >= count0:
            prediction = 1
        else:
            prediction = 0
        if prediction == classLabel[count]:
            accuracy += 1
        count += 1
    zeroOneLoss = 1 - (accuracy/count)
    if rf == True:
        print("ZERO-ONE-LOSS-RF", zeroOneLoss)
    else:
         print("ZERO-ONE-LOSS-BT", zeroOneLoss)
    return zeroOneLoss 
           
def boosting(train, test, max_depth, num_of_trees, rf, bst):
    shape = train.shape
    test_shape = test.shape
    classLabel = test[:,test_shape[1] - 2]
    training_eval = train[:,shape[1] - 2]
    roots = []
    votes = []
    accuracy = 0
    count = 0
    weights = np.full(shape[0], 1/(shape[0]), dtype=float)
    train[:, (shape[1] - 1)] = weights
    for i in range(num_of_trees):
        #print("hrtererr")
        error = 0.0
        root = build_tree(train, max_depth, rf, bst)
        roots.append(root)
        #print(root)
        errors = []
        j = 0
        error = 0
        for row in train:
            preds = predict(root, row)
            if preds != training_eval[j]:
                errors.append(1)
                error += weights[j]
            else:
                errors.append(0)
            j += 1
        #print(error)
        #print(errors)
        if error == 0:
            votes.append(1)
        else:        
            votes.append(0.5*math.log((1-error)/error))
        w = np.zeros(len(errors), dtype=float)
        sum_ = 0
        for k in range(len(errors)):
            if errors[k] == 1: 
                w[k] = weights[k] * math.exp(votes[i])
                sum_ += w[k]
            else: 
                w[k] = weights[k] * math.exp(-votes[i])
                sum_ += w[k]
        #print("summmm",sum_)
        
        for j in range(len(errors)):
            weights[j] = w[j]/sum_
            train[:,shape[1] - 1][j] = w[j]/sum_
        #print(weights)
        #print(train[:,shape[1] - 1])
        
    for row in test:
        prediction = 0
        predictions = [votes[l]*predict(roots[l], row) for l in range(len(votes))]
        predictions = np.array(predictions)
        pred = 0
        for m in range(len(predictions)):
            pred += predictions[m]
        if pred > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction == classLabel[count]:
            accuracy += 1
        count += 1
        
    zeroOneLoss = 1 - (accuracy/count)
    print("ZERO-ONE-LOSS-BST", zeroOneLoss)
    return zeroOneLoss
    
            
#train = loadfile('yelp_data.csv')

#wordlist = mostfreqwords(train, 1000)
#featurevector(train, wordlist, False)
#train_table = table(train)
#bst_train = table_boosting(train)
#decisionTree(train_table, train_table, 10,False, False)
#roots = bagging(train_table, train_table, 10,50, True, False)
#roots = boosting(bst_train, bst_train, 10,2, False, True)
if (len(sys.argv) == 4):
    trainingData = sys.argv[1]
    testData = sys.argv[2] 
    train = loadfile(trainingData)
    test = loadfile(testData)
    wordlist = mostfreqwords(train, 1000)
    featurevector(train, wordlist,False)
    featurevector(test, wordlist, False)
    train_table = table(train)
    test_table = table(test) 
    if(sys.argv[3] == '1'):
        decisionTree(train_table, test_table, 10,False, False)
    elif(sys.argv[3] == '2'):
        bagging(train_table, test_table, 10,50, False, False)
    elif(sys.argv[3] == '3'):
        bagging(train_table, test_table, 10,50, True, False)
            
   
else:
    print("3 arguments needed. Invalid input!!")
    exit()


def datapartition(document):
    percent =  [0.025, 0.05, 0.125, 0.25]
    data = {}
    keys=list(document.keys())
    random.shuffle(keys)
    count = 0
    for k in range(10):
        data[k] = {}
        for i in range(200):
            data[k][keys[count]]=document[keys[count]]
            count += 1
    std_misclassification_dt = []
    std_misclassification_svm = []
    std_misclassification_rf = []
    avg_misclassification_rf = []
    avg_misclassification_svm = []
    avg_misclassification_dt = []
    std_misclassification_bt = []
    avg_misclassification_bt = []
    std_misclassification_bst = []
    avg_misclassification_bst = []
    for j in percent:
            size=math.ceil(j*2000)
            val_dt = []
            val_svm = []
            val_rf = []
            val_bst = []
            val_bt = []
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
                wordlist = mostfreqwords(train, 1000)
                temp =list(data[l].keys())
                for i in range(200):
                    test[temp[i]] = {}
                    test[temp[i]] = [data[l][temp[i]][0],data[l][temp[i]][1]]
                for v in range(5):    
                    if(v == 0):    
                        featurevector(train, wordlist,False)
                        featurevector(test, wordlist, False)
                        train_table = table(train)
                        test_table = table(test)
                        val = decisionTree(train_table, test_table, 10,False, False)
                        val_dt.append(val)
                    if(v == 1):
                        featurevector(train, wordlist,False)
                        featurevector(test, wordlist, False)
                        X = xValues(train)
                        classLabel = yValues(train)
                        shape = X.shape[1]
                        coeff = np.zeros(shape)
                        coeff = gradient_descent_svm(classLabel, X, coeff)
                        X_test = xValues(test)
                        classLabel_test = yValues(test)
                        val = prediction_svm(coeff, X_test, classLabel_test)
                        val_svm.append(val)
                    if(v == 2):
                        featurevector(train, wordlist,False)
                        featurevector(test, wordlist, False)
                        train_table = table(train)
                        test_table = table(test)
                        val = bagging(train_table, test_table, 10,50,True, False)
                        val_rf.append(val)
                    if(v == 3):
                        featurevector(train, wordlist,False)
                        featurevector(test, wordlist, False)
                        train_table = table(train)
                        test_table = table(test)
                        val = bagging(train_table, test_table, 10,50,False, False)
                        val_bt.append(val)
                    if(v == 4):
                        featurevector(train, wordlist,False)
                        featurevector(test, wordlist, False)
                        train_table = table_boosting(train)
                        test_table = table_boosting(test)
                        val = boosting(train_table, test_table, 10,50,False, True)
                        val_bst.append(val)
                        
            avg_misclassification_dt.append(np.average(val_dt))
            avg_misclassification_svm.append(np.average(val_svm))
            avg_misclassification_rf.append(np.average(val_rf))
            avg_misclassification_bt.append(np.average(val_bt))
            avg_misclassification_bst.append(np.average(val_bst))
            std_misclassification_dt.append(np.std(val_dt)/math.sqrt(10))
            std_misclassification_svm.append(np.std(val_svm)/math.sqrt(10))
            std_misclassification_rf.append(np.std(val_rf)/math.sqrt(10))
            std_misclassification_bt.append(np.std(val_bt)/math.sqrt(10))
            std_misclassification_bst.append(np.std(val_bst)/math.sqrt(10))
    grp.figure(1)
    grp.errorbar(percent, avg_misclassification_dt, std_misclassification_dt,  marker='^',  label = "DT 0-1 loss")
    grp.errorbar(percent, avg_misclassification_rf, std_misclassification_rf,  marker='^',  label = "RF 0-1 loss")
    grp.errorbar(percent, avg_misclassification_bt, std_misclassification_bt,  marker='^',  label = "BT 0-1 loss")
    grp.errorbar(percent, avg_misclassification_svm, std_misclassification_svm,  marker='^',  label = "SVM 0-1 loss")
    grp.errorbar(percent, avg_misclassification_bst, std_misclassification_bst,  marker='^',  label = "BST 0-1 loss")
    grp.xlabel('% of TSS')
    grp.ylabel('Zero-One Loss')
    grp.legend()
    grp.savefig('HW4_TSS_1.png')
    master = []
    master.append(avg_misclassification_dt)
    master.append(avg_misclassification_svm)
    master.append(avg_misclassification_rf)
    master.append(avg_misclassification_bt)
    master.append(avg_misclassification_bst)
    master.append(std_misclassification_dt)
    master.append(std_misclassification_svm)
    master.append(std_misclassification_rf)
    master.append(std_misclassification_bt)
    master.append(std_misclassification_bst) 
    return master

#masterlist = datapartition(train)
#print(masterlist)
 
