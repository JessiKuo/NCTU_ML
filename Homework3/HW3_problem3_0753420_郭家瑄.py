# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 01:28:24 2018

@author: Kuo
"""
from PIL import Image
import numpy as np
from prettytable import PrettyTable
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def log_likelihood(p):
    return np.sum(np.log(np.sum(p, axis=1)))

if __name__ == '__main__':
    img = Image.open('./problem3/hw3.jpg')
    img.load()
    
    data = np.asarray(img, dtype='float')/255
    m, n, l = data.shape
    data = np.reshape(data, (-1, l))
    max_iter = 100
    kSet = [2, 3, 5, 20]
    
    for kk in range(len(kSet)):
        k = kSet[kk]
        total_length = m * n
    
        #k-means
        u = np.random.rand(k, l)
        r = np.full([total_length], k + 1)
    
        for i in range(300):
            dist = np.sum((data[:, None] - u) ** 2, axis=2)
            new_r = np.argmin(dist, axis=1)
    
            if np.array_equal(r, new_r):
                break
            else:
                r = new_r
    
            for j in range(k):
                data_k = data[np.where(r == j)]
                if len(data_k) == 0:
                    u[j] = np.random.rand(l)
                else:
                    u[j] = np.mean(data_k, axis=0)
                    
        table = PrettyTable()
        table.add_column("k-means mean value", range(k))
        table.add_column("r", np.round(u[:, 0]*255).astype('int'))
        table.add_column("g", np.round(u[:, 1]*255).astype('int'))
        table.add_column("b", np.round(u[:, 2]*255).astype('int'))
        print(table)
        
        #GMM
        pi = np.array([len(np.where(r == i)[0])/float(total_length) for i in range(k)])
        cov = np.array([np.cov(data[np.where(r == i)].T) for i in range(k)])
        psb = np.array([multivariate_normal.pdf(data, mean=u[i], cov=cov[i]) for i in range(k)]).T * pi
    
        likelihood = []
        likelihood.append(log_likelihood(psb))
    
        for i in range(max_iter):
            #E step
            r = psb/np.sum(psb, axis=1)[:, None]
    
            #M step
            N = np.sum(r, axis=0)
            u = np.sum(data[:, None] * r[:, :, None], axis=0)/N[:, None]
            for j in range(k):
                cov[j] = ((data - u[j]) * r[:, j, None]).T.dot(data - u[j])/N[j]
            pi = N/total_length
    
            #evaluate
            for j in range(k):
                try:
                    psb[:, j] = multivariate_normal.pdf(data, mean=u[j], cov=cov[j])*pi[j]
                except np.linalg.linalg.LinAlgError:
                    u[j] = np.random.rand(l)
                    temp = np.random.rand(l, l)
                    cov[j] = temp.dot(temp.T)
                    psb[:, j] = multivariate_normal.pdf(data, mean=u[j], cov=cov[j])*pi[j]
    
            likelihood.append(log_likelihood(psb))
        
        
        plt.plot(likelihood)
        plt.title('log likelihood curve of GMM')
        plt.xlabel('iterations')
        plt.ylabel('log p(x)')
        plt.show()
        
        r = np.argmax(psb, axis=1)
        new_data = np.round(u[r]*255)
        disp2 = Image.fromarray(new_data.reshape(m, n, l).astype('uint8'))
#        disp2.show(title='GMM')
        disp2.save('GMM_K='+str(k)+'.png')