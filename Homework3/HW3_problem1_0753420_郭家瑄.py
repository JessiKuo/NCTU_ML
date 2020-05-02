# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:40:03 2018

@author    : Kuo
@title     : machine learning (homework 3)
@problem 1 : Gaussian Prcoess for regression
@dataset   : ./problem1/gp.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prettytable

posSet = {0:(0, 0), 1:(0, 1), 2:(1, 0), 3:(1, 1)}
theta = [[1, 4, 0, 0], [0, 0, 0, 1], [1, 4, 0, 5], [1, 64, 10, 0]]
simNum = 300
trainTestSplitPnt = 60
betaInv = 1

def kernel(x1, x2, param):
    return param[0]*np.exp(-(param[1]/2)*(np.subtract.outer(x1, x2)**2))+param[2]+param[3]*np.multiply.outer(x1.T, x2)

def RMS(y, t):
    return np.sqrt(np.sum((y-t)**2)/len(t))


if __name__ == '__main__':
    data = pd.read_csv('./problem1/gp.csv', header=None)
    train_x, train_t = data.iloc[:trainTestSplitPnt:,0].values, data.iloc[:trainTestSplitPnt:,1].values
    test_x, test_t = data.iloc[trainTestSplitPnt::,0].values, data.iloc[trainTestSplitPnt::,1].values
    
    x = np.linspace(0, 2, simNum)
    y = np.zeros(simNum)
    yUpper = np.zeros(simNum)
    yLower = np.zeros(simNum)
    
    fig, ax = plt.subplots(2, 2, figsize=(15,11))
    table = prettytable.PrettyTable(['theta', 'train RMS', 'test RMS'])
    
    for i in range(len(theta)):
        pos = posSet[i]
        param = theta[i]
        CnInv = np.linalg.inv(kernel(train_x, train_x, param) + betaInv*np.identity(trainTestSplitPnt))
        
        for dataPnt in range(simNum):
            k = kernel(x[dataPnt], train_x, param)
            c = kernel(x[dataPnt], x[dataPnt], param) + betaInv
            y[dataPnt] = np.linalg.multi_dot([k.T, CnInv, train_t])
            std = np.sqrt(c - np.linalg.multi_dot([k.T, CnInv, k]))
            yUpper[dataPnt] = y[dataPnt] + std
            yLower[dataPnt] = y[dataPnt] - std
        
        train_y = np.zeros(len(train_x))
        #計算 training RMS
        for j in range(len(train_x)):
            k = kernel(train_x[j], train_x , param)
            train_y[j] = np.linalg.multi_dot([k.T, CnInv, train_t])
            
        test_y = np.zeros(len(test_x))
        #計算 testing RMS
        for j in range(len(test_x)):
            k = kernel(test_x[j], train_x, param)
            test_y[j] = np.linalg.multi_dot([k.T, CnInv, train_t])
        
        table.add_row([str(param), RMS(train_y, train_t), RMS(test_y, test_t)])
        
        ax[pos].plot(x, y, 'r-')
        ax[pos].fill_between(x, yUpper, yLower, facecolor='pink')
        ax[pos].scatter(train_x, train_t, facecolor='none', edgecolor='b')
        ax[pos].set_title('theta = '+str(param))
        ax[pos].set_xlabel('x')
        ax[pos].set_ylabel('y', rotation = 0)
        
    plt.show()
    print(table)
    
    