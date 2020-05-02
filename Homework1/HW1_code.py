# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:51:31 2018

@author: Kuo
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

def loadData(path):
    df = pd.read_csv(path)
    splitPoint = int(len(df)*0.9)
    train = df[:splitPoint]
    test = df[splitPoint:]
    return train, test

def cut(train, test):
    train_x = train.iloc[:,:3].values
    train_y = train.iloc[:,3].values
    test_x = test.iloc[:,:3].values
    test_y = test.iloc[:,3].values
    return train_x, train_y, test_x, test_y

def RMSE(pred, true):
    return np.sqrt(np.sum((pred - true) ** 2)/len(pred))

def transform(data, dim, order):
    trans_data = np.full((len(data), 1), 1)
    for i in range(order):
        for comb in list(combinations_with_replacement(range(dim), i+1)):
            p = np.prod(data[:, comb], axis=1).reshape(-1, 1)
            trans_data = np.concatenate((trans_data, p), axis=1)

    return trans_data

def getCoef(data, target):
    Xt = np.transpose(data)
    w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, data)), Xt), target)
    return w

def getCoefReg(data, target, order, lam):
    Xt = np.transpose(data)
    w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, data) + lam*np.identity(order+1)), Xt), target)
    return w

if __name__ == '__main__':
    train, test = loadData('housing.csv')
    train_x, train_y, test_x, test_y = cut(train, test)
    
    #%%
    ''''
        第一小題
    '''
    trErr = []
    teErr = []
    
    for i in range(3):
        transTrain = transform(train_x, 3, i+1)
        transTest = transform(test_x, 3, i+1)
    
        w = getCoef(transTrain, train_y)
    
        trainPred = np.dot(transTrain, w)
        testPred = np.dot(transTest, w)
    
        trainErr = RMSE(trainPred, train_y)
        testErr = RMSE(testPred, test_y)
        
        teErr.append(testErr)
        trErr.append(trainErr)
        
        print('For M = :', i+1)
        print('Training error is', trainErr, ", testing error is", testErr)
    
    
    Err = pd.DataFrame()
    Err['trErr'] = trErr
    Err['teErr'] = teErr
    Err = Err.set_index([[1,2,3]])
    
    plt.plot(Err.index, Err['teErr'], label='test')
    plt.plot(Err.index, Err['trErr'], label='train')
    plt.xlabel('M')
    plt.ylabel('RMSE')
    plt.xticks(range(1, 4))
    plt.legend()
    plt.show()
    
    
    #%%
    '''
        第二小題
    '''
    features = ['total room', 'population', 'median income']
    error_list = []
    for i in range(len(features)):
        transTrain2 = transform(np.delete(train_x, i, axis=1), 2, 3)
        
        w2 = getCoef(transTrain2, train_y)

        trainPred2 = np.dot(transTrain2, w2)

        error_list.append(RMSE(trainPred2, train_y))
        
        print('RMSE after remove feature [', features[i], '] =', error_list[len(error_list)-1])

    print('The most contributive attribute is "', features[error_list.index(max(error_list))], '"') 
    
    
    
    #%%
    '''
        第三小題
    '''
    lam = [0.1, 0.001]
    trErr = []
    teErr = []
    
    for j in range(len(lam)):
        tmp = 0
        tmpTrErr = []
        tmpTeErr = []
        for i in range(3):
            transTrain = transform(train_x, 3, i+1)
            transTest = transform(test_x, 3, i+1)
        
            tmp += len(list(combinations_with_replacement(range(3), i+1)))
            
            Xt = np.transpose(transTrain)
            w = np.dot(np.dot(np.linalg.inv(np.dot(Xt, transTrain) + lam[j]*np.identity(tmp+1)), Xt), train_y)
        
            trainPred = np.dot(transTrain, w)
            testPred = np.dot(transTest, w)
        
            trainErr = RMSE(trainPred, train_y)
            testErr = RMSE(testPred, test_y)
            
            tmpTeErr.append(testErr)
            tmpTrErr.append(trainErr)
            
            print('For M = :', i+1)
            print('Training error is', trainErr, ", testing error is", testErr)
        trErr.append(tmpTrErr)
        teErr.append(tmpTeErr)
     
    # lambda = 0.1
    Err = pd.DataFrame()
    Err['trErr'] = trErr[0]
    Err['teErr'] = teErr[0]
    Err = Err.set_index([[1,2,3]])
    
    plt.plot(Err.index, Err['teErr'], label='test, lambda=0.1')
    plt.plot(Err.index, Err['trErr'], label='train, lambda=0.1')
    plt.xlabel('M')
    plt.ylabel('RMSE')

    # lambda = 0.1
    Err = pd.DataFrame()
    Err['trErr'] = trErr[1]
    Err['teErr'] = teErr[1]
    Err = Err.set_index([[1,2,3]])
    
    plt.plot(Err.index, Err['teErr'], label='test, lambda=0.001')
    plt.plot(Err.index, Err['trErr'], label='train, lambda=0.001')
    plt.xlabel('M')
    plt.ylabel('RMSE')
    plt.xticks(range(1, 4))
    plt.legend()
    plt.show()        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
