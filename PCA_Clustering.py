# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:25:49 2017

@author: Shubhika
"""
import numpy as np
import math
import scipy 
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sys
import matplotlib

def loadfile(filename):
    data = np.genfromtxt( filename, delimiter=',')
    return data

def init_centroids(data, k):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]
    return centroids

def clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters, distances 

def new_centroids(data, clusters, centroids):
    centroids = np.array([data[clusters==i].mean(axis=0) for i in range(centroids.shape[0])])
    return centroids


def kmeans(data, k, y):
    centroids = init_centroids(data, k)
    cluster = None
    distances = None
    for i in range(50):
        cluster, distances = clusters(data, centroids)
        centroids = new_centroids(data, cluster, centroids)
    print("WS_SSD: ", wc_ssd(centroids, cluster, data))
    print("SC: ",sc(centroids,cluster, data))
    print("NMI: ",nmi(y, cluster))
    #return  wc_ssd(centroids, cluster, data), sc(centroids,cluster, data)
    #return cluster

def wc_ssd(centroids, cluster, data):
    sum1 = 0
    for i in range(centroids.shape[0]):
        for j in range(data.shape[0]):
            if i == cluster[j]:
                sum1 += ((data[j][0] - centroids[i][0])**2 +  (data[j][1] - centroids[i][1])**2)
    return sum1

def sc(centroids, cluster, data):
    Y = distance.cdist(data, data, 'euclidean')
    center = distance.cdist(data,centroids)
    i = 0
    A = []
    B = []
    for i in range(len(data)):
        cluster_no = cluster[i]
        A.append(np.mean([Y[i][j] for j in range(len(data)) if((not i==j) and cluster[i] == cluster[j])]))
        closest_cluster = 0
        min_val = 1000000
        for k in range(len(centroids)):
            if not cluster_no == k:
                dis = center[i,k]
                if(dis < min_val):
                    closest_cluster = k
                    min_val = dis
        B.append(np.mean([Y[i][j] for j in range(len(data)) if((not i==j) and (cluster[j] == closest_cluster))]))
    A = np.array(A)
    B = np.array(B)
    maxm = np.maximum(A,B)
    #print((B-A)/maxm)
    sc = np.mean((B-A)/maxm)
    #print(sc)
    return sc

def nmi(y, cluster):
    n_cluster = set(cluster)
    n_y = set(y)
    N = len(y)
    i_val = 0.0
    hw_val = 0.0
    hc_val = 0.0
    for i in n_cluster:
        for j in n_y:
            wk_cj= 0.0
            wk = 0.0
            cj = 0.0
            for k in range(len(y)):
                if cluster[k] == i:
                       wk += 1
                if y[k] == j:
                    cj += 1
                    if cluster[k] == i:
                       wk_cj += 1
            if not wk_cj == 0:
                i_val += ((wk_cj)/N)*math.log((N*wk_cj)/(wk*cj))
        hw_val += (wk/N)*math.log(wk/N)
    for j in n_y:
        cj = 0.0
        for k in range(len(y)):
            if y[k] == j:
                cj += 1
        hc_val += (cj/N)*math.log(cj/N)
    nmi = i_val/(-hc_val -hw_val)
    return nmi

#performing principal component analysis 
def pca(data_pca):
    mean_vectors = np.mean(data_pca, axis = 0)
    data_std = data_pca - mean_vectors
    cov_mat = np.cov(np.transpose(data_std))
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals = np.abs(eig_vals)
    eig_vecs[eig_vals.argsort()]
    eig_vecs = eig_vecs[:,:10]
    eig_vecs = np.real(eig_vecs)
    Y = data_std.dot(eig_vecs)
    return Y



if(len(sys.argv) == 3):
    filename = sys.argv[1]
    data = loadfile(filename)
    y = data[:,1]
    kmeans(data[:,[2,3]], int(sys.argv[2]),y)
else:
    print('error!!!')


#experiments regaring word counts and cluster sizes 

def exp(data):
    k_val = [2, 4, 8, 12, 25]
    wc_data1 = []
    wc_data2 = []
    wc_data3 = []
    sc_data1 = []
    sc_data2 = []
    sc_data3 = []
    for k in k_val:
        for i in range(3):
            i = 2
            centroid = np.zeros(shape=(k,2))
            if(i == 0):
                clusters = fcluster(linkage(data[:,[2,3]], method='complete'),16, criterion='maxclust') - 1
                nmi(data[:,1], clusters)
                break
                centroid = new_centroids(data, clusters, centroid)
                sc1 = sc(centroid, clusters, data)
                wc = wc_ssd(centroid, clusters, data)
                wc_data1.append(wc)
                sc_data1.append(sc1)
            elif(i == 1):
                clusters = fcluster(linkage(data[:,[2,3]], method='average'),16,criterion='maxclust') - 1
                nmi(data[:,1], clusters)
                break                   
                centroid = new_centroids(data, clusters, centroid)
                sc1 = sc(centroid, clusters, data)
                wc = wc_ssd(centroid, clusters, data)
                wc_data2.append(wc)
                sc_data2.append(sc1)
            elif(i == 2):
                clusters = fcluster(linkage(data[:,[2,3]], method='single'),23,criterion='maxclust') - 1
                centroid = new_centroids(data, clusters, centroid)
                nmi(data[:,1], clusters)
                break
                sc1 = sc(centroid, clusters, data)
                wc = wc_ssd(centroid, clusters, data)
                wc_data3.append(wc)
                sc_data3.append(sc1)
    '''#plt.figure(1)
    #plt.errorbar(k, wc_data1,  marker='^',  label = "Complete WC_SSD")
    #plt.errorbar(k, wc_data2, marker='^',  label = "Average WC_SSD")
    plt.errorbar(k, wc_data3,  marker='^',  label = "Single WC_SSD")
    plt.errorbar(k, sc_data1, marker='^',  label = "Complete SC")
    plt.errorbar(k, sc_data2, marker='^',  label = "Average SC")
    plt.errorbar(k, sc_data3, marker='^',  label = "Single SC")
    plt.xlabel('k value')
    plt.ylabel('wc_ssd/sc')
    plt.legend()
    plt.savefig('HW5_B_1.png')'''
    masterlist = []
    masterlist.append(wc_data1)
    masterlist.append(wc_data2)
    masterlist.append(wc_data3)
    masterlist.append(sc_data1)
    masterlist.append(sc_data2)
    masterlist.append(sc_data3)
    print(masterlist)
    return masterlist    

def loadfile_2(filename):
    data = np.genfromtxt( filename, delimiter=',')
    data1 = data[data[:,1] == 2]
    data2 = data[data[:,1] == 4]
    data3 = data[data[:,1] == 6]
    data4 = data[data[:,1] == 7]
    data = np.concatenate((data1, data2, data3, data4))
    return data

def loadfile_3(filename):
    data = np.genfromtxt( filename, delimiter=',')
    data3 = data[data[:,1] == 6]
    data4 = data[data[:,1] == 7]
    data = np.concatenate((data3, data4))
    return data

def questionB(filename):
    data1 = loadfile(filename)
    data2 = loadfile_2(filename)
    data3 = loadfile_3(filename)
    k = [2, 4, 8, 16, 32]
    avg_wc_data1 = []
    avg_wc_data2 = []
    avg_wc_data3 = []
    avg_sc_data1 = []
    avg_sc_data2 = []
    avg_sc_data3 = []
    std_wc_data1 = []
    std_wc_data2 = []
    std_wc_data3 = []
    std_sc_data1 = []
    std_sc_data2 = []
    std_sc_data3 = []
    for val in k:
        for j in range(10):
            wc_data1 = []
            wc_data2 = []
            wc_data3 = []
            sc_data1 = []
            sc_data2 = []
            sc_data3 = []
            print(val)
            for i in range(3):
                print(i)
                if(i == 0):
                    wc, sc = kmeans(data1[:,[2,3]], val, data1[:,1])
                    wc_data1.append(wc)
                    sc_data1.append(sc)
                elif(i == 1):
                    wc, sc = kmeans(data2[:,[2,3]], val, data2[:,1])
                    wc_data2.append(wc)
                    sc_data2.append(sc)
                elif(i == 2):
                    wc, sc = kmeans(data3[:,[2,3]], val, data3[:,1])
                    wc_data3.append(wc)
                    sc_data3.append(sc)
        avg_wc_data1.append(np.mean(wc_data1))
        avg_wc_data2.append(np.mean(wc_data2))
        avg_wc_data3.append(np.mean(wc_data3))
        avg_sc_data1.append(np.mean(sc_data1))
        avg_sc_data2.append(np.mean(sc_data2))
        avg_sc_data3.append(np.mean(sc_data3))
        std_wc_data1.append(np.std(wc_data1))
        std_wc_data2.append(np.std(wc_data2))
        std_wc_data3.append(np.std(wc_data3))
        std_sc_data1.append(np.std(sc_data1))
        std_sc_data2.append(np.std(sc_data2))
        std_sc_data3.append(np.std(sc_data3))
    masterlist = []
    masterlist.append(avg_wc_data1)
    masterlist.append(avg_wc_data2)
    masterlist.append(avg_wc_data3)
    masterlist.append(avg_sc_data1)
    masterlist.append(avg_sc_data2)
    masterlist.append(avg_sc_data3)
    masterlist.append(std_wc_data1)
    masterlist.append(std_wc_data2)
    masterlist.append(std_wc_data3)
    masterlist.append(std_sc_data1)
    masterlist.append(std_sc_data2)
    masterlist.append(std_sc_data3)
    print(masterlist)
    return masterlist


def clusters_3(data):
    Z = linkage(data, 'single')
    plt.figure(figsize=(25, 10))
    plt.title('Single Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis label
    )
    plt.show()
    
def loadfile_junk(filename):
    data = np.genfromtxt( filename, delimiter=',')
    data0 = data[data[:,1] == 0]
    np.random.shuffle(data0)
    data0 = data0[:10,:]
    data1 = data[data[:,1] == 1]
    np.random.shuffle(data1)
    data1 = data1[:10,:]
    data2 = data[data[:,1] == 2]
    np.random.shuffle(data2)
    data2 = data2[:10,:]
    data3 = data[data[:,1] == 3]
    np.random.shuffle(data3)
    data3 = data3[:10,:]
    data4 = data[data[:,1] == 4]
    np.random.shuffle(data4)
    data4 = data4[:10,:]
    data5 = data[data[:,1] == 5]
    np.random.shuffle(data5)
    data5 = data5[:10,:]
    data6 = data[data[:,1] == 6]
    np.random.shuffle(data6)
    data6 = data6[:10,:]
    data7 = data[data[:,1] == 7]
    np.random.shuffle(data7)
    data7 = data7[:10,:]
    data8 = data[data[:,1] == 8]
    np.random.shuffle(data8)
    data8 = data8[:10,:]
    data9 = data[data[:,1] == 9]
    np.random.shuffle(data9)
    data9 = data9[:10,:]
    data = np.concatenate((data1, data2, data3, data4,data5, data6, data7, data8, data9, data0))
    return data

    def print_image(row):
    fix,ax = plt.subplots(nrows = 2, ncols = 5,sharex = True, sharey = True,)
    ax = ax.flatten()
    for i in range(10):
        column = row[:,i]
        pixels = column.reshape((28,28))
        ax[i].imshow(pixels, cmap='gray')     
        
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def lists(data, y):
    for j in range(len(data)):
        if 0.0 == y[j]:
            #print(y[j])
            print_image(data[j])
            break
def print_2D(inputData,N=1000, seed=123):
    np.random.seed(seed)
    marker_list = ['<','s','o','^','*','v','1','2','3','4']
    colors = ['lightblue','pink','aqua','c','lightyellow','y','k','lightskyblue','crimson','olive']
    newMatrix = inputData[np.random.choice(inputData.shape[0], N, replace=False), :]
    label = np.unique(newMatrix[:,10]).astype(int)
    plt.scatter(newMatrix[:,0], newMatrix[:,1], c=newMatrix[:,10], cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(label)
    plt.show()


def questionB(filename):
    data1 = loadfile(filename)
    data1 = pca(data1[:,2:])
    data2 = loadfile_2(filename)
    data2 = pca(data2[:,2:])
    data3 = loadfile_3(filename)
    data3 = pca(data3[:,2:])
    k = [2, 4, 8, 16, 32]
    wc_data1 = []
    wc_data2 = []
    wc_data3 = []
    sc_data1 = []
    sc_data2 = []
    sc_data3 = []
    for val in k:
        print(val)
        for i in range(3):
            print(i)
            #i = 0
            if(i == 0):
                wc,sc = kmeans(data1[:,[0,1]], 8, data1[:,1])
                #print_2D(data1[:,[2,3]],clusters)
                wc_data1.append(wc)
                sc_data1.append(sc)
            elif(i == 1):
                #print("here")
                wc, sc = kmeans(data2[:,[0,1]], 4, data2[:,1])
                wc_data2.append(wc)
                sc_data2.append(sc)
            elif(i == 2):
                #print("here")
                wc, sc = kmeans(data3[:,[0,1]], 2, data3[:,1])
                wc_data3.append(wc)
                sc_data3.append(sc)
    '''plt.figure(1)
    plt.errorbar(k, wc_data1,  marker='^',  label = "D1 WC_SSD")
    plt.errorbar(k, wc_data2, marker='^',  label = "D2 WC_SSD")
    plt.errorbar(k, wc_data3,  marker='^',  label = "D3 WC_SSD")
    plt.errorbar(k, sc_data1, marker='^',  label = "D1 SC")
    plt.errorbar(k, sc_data2, marker='^',  label = "D2 SC")
    plt.errorbar(k, sc_data3, marker='^',  label = "D3 SC")
    plt.xlabel('k value')
    plt.ylabel('wc_ssd/sc')
    plt.legend()
    plt.savefig('HW5_B_1.png')'''
    masterlist = []
    masterlist.append(wc_data1)
    masterlist.append(wc_data2)
    masterlist.append(wc_data3)
    masterlist.append(sc_data1)
    masterlist.append(sc_data2)
    masterlist.append(sc_data3)
    print(masterlist)
    return masterlist