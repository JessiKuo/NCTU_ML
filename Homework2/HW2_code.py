# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:31:03 2018

@author: Kuo
"""
#%%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from itertools import combinations
from prettytable import PrettyTable
#%%
Pos = {'10':(0, 0), '15':(0, 1), '30':(1, 0), '80':(1, 1)}

def sigmoid(a):
    return 1./(1 + math.exp(-a))

def getPhi(x, s, M):
    BigPhi = np.empty([len(x), M])
    mu = [(4*j)/M for j in range(M)]
    for i in range(len(x)):
        BigPhi[i] = np.array([sigmoid((x[i]-mu[j])/s) for j in range(M)])
    return BigPhi

def getMnSn(beta, alpha, BigPhi, t):    
    Sn = np.linalg.inv(alpha*np.identity(len(BigPhi[0])) + beta*np.dot(BigPhi.T, BigPhi))
    Mn = beta*np.linalg.multi_dot([Sn, BigPhi.T, t])
    return Mn, Sn

def Entropy(train_y, y):
    return -np.sum(train_y * np.log(y))

def evaluate(train_y, y):
    yi = np.argmax(y, axis=1)
    train_yi = np.argmax(train_y, axis=1)
    return np.sum(yi==train_yi)/len(yi)
    
def wGradient(train_y, y, train_x):
    return np.dot((y-train_y).T, train_x)

def Hessian(y, train_x):
    y2 = y.T
    R = np.zeros((y.shape[0], y.shape[0]))
    for j in range(0, y.shape[1]): # 3classes
        for k in range(0, y.shape[0]): #148 train entry
            R[k,k] = y2[j,k]*(1 - y2[j,k])
        H = train_x.T.dot(R).dot(train_x)    
    return H

def find_border(t):
    b = []
    for i in range(1, len(t)):
        if np.any(t[i, :] - t[i-1, :]):
            b.append(i)
    return b

def plot(axarra, b, c1, c2, c3, variable):
    bins = np.linspace(b[0], b[1], 10)
    axarra.set_title('Variable {}'.format(variable))
    axarra.hist(c1, bins=bins, alpha=0.5, label='class 1')
    axarra.hist(c2, bins=bins, alpha=0.5, label='class 2')
    axarra.hist(c3, bins=bins, alpha=0.5, label='class 3')
    axarra.legend(loc='upper right')

#%%
if __name__ == '__main__':
# =============================================================================
#     Q1
# =============================================================================
    data = pd.read_csv('1_data.csv')
    x, t = data['x'], data['t']
    M = 7
    s = 0.1
    N = [10, 15, 30, 80]
    beta = 1
    alpha = 1./(10**6)
    
    f, axs = plt.subplots(2, 2,figsize=(15,10))
    f2, axs2 = plt.subplots(2, 2,figsize=(15,10))
    
    xx = np.arange(0, 4, 0.004)
    xPhi = getPhi(xx, s, M)
    
    y2 = np.empty(len(xx))
    y2_upper = np.empty(len(xx))
    y2_lower = np.empty(len(xx))
    
    for n in range(len(N)):
        tMn = PrettyTable()
        tMn.field_names = ['Mn']
        
        tSn = PrettyTable()
        tSn.field_names = ['Sn']
        
        figPos = Pos[str(N[n])]
        
        BigPhi = getPhi(x[:N[n]], s, M)
        Mn, Sn = getMnSn(beta, alpha, BigPhi, t[:N[n]])
        
        for i in range(len(Mn)):
            tMn.add_row([np.round(Mn[i], 2)])
        
        print(tMn)
        for i in range(len(Sn)):
            tSn.add_row([np.round(Sn[i], 2)])
        print(tSn)
        
        for l in range(5):
            w = np.random.multivariate_normal(Mn, Sn)
            y = np.dot(w, xPhi.T)
            axs[figPos].plot(xx, y, 'r-')

        for i in range(len(xx)):
            Phi = getPhi(np.array([xx[i]]), s, M)
            y2[i] = np.dot(Mn, Phi.T)
            var = 1./beta + np.linalg.multi_dot([Phi, Sn, Phi.T])
            std = np.sqrt(var)
            y2_upper[i] = y2[i] + std
            y2_lower[i] = y2[i] - std
        
        axs[figPos].scatter(x[:N[n]], t[:N[n]], s = 80, facecolors='none', edgecolors='b')
        axs[figPos].set_title('N = {}'.format(N[n]))
        axs[figPos].set_xlim(-0.1, 4.1)
        axs[figPos].set_ylim(-15, 15)
        axs[figPos].set_xlabel('x')
        axs[figPos].set_ylabel('t')
        
        axs2[figPos].plot(xx, y2, 'r-')
        axs2[figPos].fill_between(xx, y2_upper, y2_lower, facecolor='pink', edgecolor='none')
        axs2[figPos].scatter(x[:N[n]], t[:N[n]], s=80, facecolors='none', edgecolors='b')
        axs2[figPos].set_title('N = {}'.format(N[n]))
        axs2[figPos].set_xlim(-0.1, 4.1)
        axs2[figPos].set_ylim(-15, 15)
        axs2[figPos].set_xlabel('x')
        axs2[figPos].set_ylabel('t')
    
#%%
# =============================================================================
#     Q2
# =============================================================================
#(1)
    train = pd.read_csv('./train.csv',header = None).values
    test = pd.read_csv('./test.csv',header = None).values
    
    train_y, train_x = np.split(train, [3], axis = 1)
    
    w = np.zeros([train_y.shape[1], train_x.shape[1]])
    
    acc = []
    entropy = []
    wParam = []
    stop = 10
    
    while True:
        wParam.append(copy.deepcopy(w))
        a = np.dot(train_x, w.T)
        y = np.array([[np.exp(a[i][j])/np.sum(np.exp(a[i,:])) for j in range(3)] for i in range(len(a))])
        
        entropy.append(Entropy(train_y, y))
        acc.append(evaluate(train_y, y))
        
        if entropy[-1] < stop:
            break
        
        if math.isnan(entropy[-1]):
            entropy.pop()
            acc.pop()
            wParam.pop()
            break
        
        E = wGradient(train_y, y, train_x)
        H = Hessian(y, train_x)
        
        w -= np.dot(E, np.linalg.inv(H))*0.1
        
    f, axs = plt.subplots(2, 1, sharex='col', figsize=(10, 8))
    
    axs[0].plot(acc)
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Acc')
    axs[1].plot(entropy)
    axs[1].set_title('Cross Entropy')
    axs[1].set_xlabel('Epoch Number')
    axs[1].set_ylabel('Loss')
    plt.show()

#%%
#(2)    
    aTest = np.dot(test, wParam[-1].T)
    yTest = np.array([[np.exp(aTest[i][j])/np.sum(np.exp(aTest[i,:])) for j in range(3)] for i in range(len(aTest))])    
    
    t = PrettyTable()
    t.field_names = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'predict']
    ans = np.argmax(yTest, axis=1)
    for i in range(len(yTest)):
        t.add_row([round(test[i, 0],2), round(test[i, 1],2), round(test[i, 2],2), \
                   round(test[i, 3],2), round(test[i, 4],2), round(test[i, 5],2), round(test[i, 6],2), ans[i]])
    print(t)

#%%    
#(3)
    min_max = np.vstack([np.min(train_x, axis=0), np.max(train_x, axis=0)]).T
    border = find_border(train_y)
    [class1, class2, class3] = np.split(train_x, border)
    
    varSize = train_x.shape[1]
    
    for i in range(0, varSize, 4):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        if i < varSize:
            plot(ax[0, 0], min_max[i], class1[:, i], class2[:, i], class3[:, i], i)
        if i+1 < varSize:
            plot(ax[0, 1], min_max[i+1], class1[:, i+1], class2[:, i+1], class3[:, i+1], i+1)
        if i+2 < varSize:
            plot(ax[1, 0], min_max[i+2], class1[:, i+2], class2[:, i+2], class3[:, i+2], i+2)
        if i+3 < varSize:
            plot(ax[1, 1], min_max[i+3], class1[:, i+3], class2[:, i+3], class3[:, i+3], i+3)
        plt.show()
    
#%%    
#(5)
    pairs = list(combinations(range(train_x.shape[1]), 2))
    entropyMin = 500
    entropy = []
    
    for var in pairs:
        w = np.zeros([train_y.shape[1], 2])
        NewTrain_x = train_x[:, var]
        
        for i in range(1000):
            a = np.dot(NewTrain_x, w.T)
            y = np.array([[np.exp(a[i][j])/np.sum(np.exp(a[i,:])) for j in range(3)] for i in range(len(a))])
            
            entroTmp = Entropy(train_y ,y)
            
            if math.isnan(entroTmp):
                entropy.pop()
                break
            elif entroTmp < entropyMin:
                entropyMin = entroTmp
                final_var = var
            
            E = wGradient(train_y, y, NewTrain_x)
            H = Hessian(y, NewTrain_x)
            
            w -= np.dot(E, np.linalg.inv(H))*0.01
    
    border = find_border(train_y)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(train_x[:border[0], final_var[0]], 
                train_x[:border[0], final_var[1]], s=60, color='b', label='class 1')
    plt.scatter(train_x[border[0]:border[1], final_var[0]], 
                train_x[border[0]:border[1], final_var[1]], s=60, color='g', label='class 2')
    plt.scatter(train_x[border[1]:-1, final_var[0]], 
                train_x[border[1]:-1, final_var[1]], s=60, color='r', label='class 3')
    plt.xlabel('Variable {}'.format(final_var[0]))
    plt.ylabel('Variable {}'.format(final_var[1]))
    plt.legend()
    plt.show()
        
#%%
#(6)
    acc = []
    entropy = []
    wParam = []
    stop = 33
    w = np.zeros([train_y.shape[1], len(final_var)])
    NewTrain_x = train_x[:, final_var]
    
    while True:
        wParam.append(copy.deepcopy(w))
        a = np.dot(NewTrain_x, w.T)
        y = np.array([[np.exp(a[i][j])/np.sum(np.exp(a[i,:])) for j in range(3)] for i in range(len(a))])
        
        entropy.append(Entropy(train_y, y))
        acc.append(evaluate(train_y, y))
        
        print(entropy[-1])
        
        if entropy[-1] < stop: break
        
        if math.isnan(entropy[-1]):
            entropy.pop()
            acc.pop()
            wParam.pop()
            break
        
        E = wGradient(train_y, y, NewTrain_x)
        H = Hessian(y, NewTrain_x)
        
        w -= np.dot	(E, np.linalg.inv(H))*0.01
        
    f, axs = plt.subplots(2, 1, sharex='col', figsize=(10, 8))
    axs[0].plot(acc)
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Acc')
    axs[1].plot(entropy)
    axs[1].set_title('Cross Entropy')
    axs[1].set_xlabel('Epoch Number')
    axs[1].set_ylabel('Loss')
    plt.show()
    
    aTest = np.dot(test[:, final_var], wParam[-1].T)
    yTest = np.array([[np.exp(aTest[i][j])/np.sum(np.exp(aTest[i,:])) for j in range(3)] for i in range(len(aTest))])    
    from prettytable import PrettyTable
    t = PrettyTable()
    t.field_names = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'predict']
    ans = np.argmax(yTest, axis=1)
    for i in range(len(yTest)):
        t.add_row([round(test[i, 0],2), round(test[i, 1],2), round(test[i, 2],2), \
                   round(test[i, 3],2), round(test[i, 4],2), round(test[i, 5],2), round(test[i, 6],2), ans[i]])
    print(t)
    
#%%
#(7)
#[ref link] https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
    np.set_printoptions(precision=4)

    mean_vectors = []
    classNum = len(train_y[0,:])
    border = find_border(train_y)
    border.insert(0, 0)
    border.append(len(train_x))
    
    for cl in range(len(border)-1):
        mean_vectors.append(np.mean(train_x[border[cl]:border[cl+1]], axis=0))
    
    Sw = np.zeros((7, 7))
    for cl, mv in zip(range(1, 4), mean_vectors):
        class_sc_mat = np.zeros((7, 7))
        for row in train_x[border[cl-1]:border[cl]]:
            row, mv = row.reshape(7,1), mv.reshape(7,1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        Sw += class_sc_mat
    
    overall_mean = np.mean(train_x, axis=0)

    Sb = np.zeros((7, 7))
    for i, mean_vec in enumerate(mean_vectors):  
        n = border[i+1]-border[i]
        mean_vec = mean_vec.reshape(7,1) 
        overall_mean = overall_mean.reshape(7,1)
        Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

#    for i in range(len(eig_vals)):
#        eigvec_sc = eig_vecs[:,i].reshape(7,1)   
        
    
    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(7,1)
        np.testing.assert_array_almost_equal(np.linalg.inv(Sw).dot(Sb).dot(eigv),
                                             eig_vals[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)

    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    W = np.hstack((eig_pairs[0][1].reshape(7,1), eig_pairs[1][1].reshape(7,1)))
    
    X_lda = train_x.dot(W)
    

    ax = plt.subplot(111)
    for label,marker,color in zip(range(0, 3),('^', 's', 'o'),('blue', 'red', 'green')):
        plt.scatter(x=X_lda[:,0].real[(border[label]):(border[label+1])],
                y=X_lda[:,1].real[(border[label]):(border[label+1])],
                marker=marker,
                color=color,
                alpha=0.5,
                label='class'+str(label)
                )

#    plt.xlabel('LD1')
#    plt.ylabel('LD2')
    
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    
    
    plt.grid()
    plt.tight_layout
    plt.show()

#%%
#(3)
    data = pd.read_csv('./seeds.csv')
    train = []
    test = []
    for i in range(0, len(data), 70):
        train.append(data.loc[i:i+49])
        test.append(data.loc[i+50:i+69])
        
    train = np.concatenate(train,axis = 0)
    test = np.concatenate(test, axis = 0)
    
    train_x, train_y = train[:,:-1], train[:,-1]
    test_x, test_y = test[:,:-1], test[:,-1]

# =============================================================================
#     先正規化再切train, test    
# =============================================================================
    All = np.concatenate((train_x, test_x), axis=0)
    All = (All-np.mean(All, axis=0))/np.std(All, axis=0)
    train_x, test_x = All[:150], All[150:]
    
# =============================================================================
#     用train資料集的mean, standard正規畫train、test資料集
# =============================================================================
#    mean = np.mean(train_x, axis=0)
#    std = np.std(train_x, axis=0) 
#    train_x = (train_x-mean)/std
#    test_x = (test_x-mean)/std
    
# =============================================================================
#     train, test資料集一起正規化
# =============================================================================
#    train_x = (train_x-np.mean(train_x, axis=0))/np.std(train_x, axis=0)
#    test_x = (test_x-np.mean(test_x, axis=0))/np.std(test_x, axis=0)
    
    from collections import Counter
    
    def KNN(test, train, target, K):
        N = train.shape[0]
        dist = np.sqrt(np.sum((np.tile(test, (N, 1))-train)**2, axis=1))
        idx = sorted(range(len(dist)), key=lambda i:dist[i])[0:K]
#        return max(set(target[idx]), key = list(target[idx]).count)
        return Counter(sorted(target[idx])).most_common(1)[0][0]
        
    
    acc = []
    N = len(test_x)
    
    for j in range(10):
        CF = np.zeros((3, 3))
        for i in range(N):
            guess = KNN(test_x[i], train_x, train_y, j+1)
            CF[int(test_y[i])-1, int(guess)-1] += 1
            
        acc.append(np.trace(CF)/np.sum(CF))
        
    plt.plot(list(np.arange(1, 11)), acc, 'g')
    plt.title('KNN:3_1')
    plt.show()
    
    #(3)_2
    def KNN2(test, train, target, V):
        N = train.shape[0]
        dist = np.sqrt(np.sum((np.tile(test, (N, 1))-train)**2, axis=1))
        idx = np.nonzero(dist < V)[0]
        return Counter(sorted(target[idx])).most_common(1)[0][0]
#        return max(set(target[idx]), key = list(target[idx]).count)
    
    acc2 = []
    cc = []
    N = len(test_x)
    cnt = 0
    for v in range(2, 11):
        CF = np.zeros((3, 3))
        for i in range(N):
            guess = KNN2(test_x[i], train_x, train_y, v)                         
            CF[int(test_y[i])-1, int(guess)-1] += 1
        acc2.append(np.trace(CF)/np.sum(CF))
        cc.append(CF)
    plt.plot(list(np.arange(2, 11)), acc2, 'g')
    plt.title('KNN:3_2')
    plt.show()
    
    
    
    
    
    
