import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
K_init = len(means)
cov = [[1, 0], [0, 1]]
N = 100
K = 5
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# Vẽ dữ liệu X0 với màu đỏ, X1 với màu xanh lá cây, X2 với màu xanh lam


X = np.concatenate((X0, X1, X2), axis=0)
random_indices = np.random.choice(len(X), K, replace=False)
M = X[random_indices]
Y = np.zeros((N*K_init, K))
def init_centroids(X, K):
    return X[np.random.choice(len(X), K, replace = False)]

def distance(a, b):
    res = 0
    for i in range(len(a)):
        res += (a[i] - b[i])**2
    return res

def assign_cluster(X, Y, M, N, K):
    for i in range(N*K_init):
        minDis = 10e10
        for j in range(K):
            Y[i][j] = 0
            if distance(X[i], M[j]) < minDis:
                res = j
                minDis = distance(X[i], M[j])
        Y[i][res] = 1
    return Y

def assign_centroids(X, K, N):
    newM = np.zeros((K, len(X[0])))
    for i in range(K):
        sum = 0
        for j in range(N*K_init):
            if Y[j][i] == 1:
                newM[i] += X[j]
                sum += 1
        newM[i] /= sum
    return newM

def showResult(X, Y, M, loop):
    plt.figure(1)
    plt.scatter(X0[:, 0], X0[:, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X1[:, 0], X1[:, 1], c='green', marker='s', label='Class 1')
    plt.scatter(X2[:, 0], X2[:, 1], c='blue', marker='^', label='Class 2')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best')

    plt.figure(2)
    X_class1 = X[Y[:, 0] == 1]
    X_class2 = X[Y[:, 1] == 1]
    X_class3 = X[Y[:, 2] == 1]
    X_class4 = X[Y[:, 3] == 1]
    X_class5 = X[Y[:, 4] == 1]
    plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X_class2[:, 0], X_class2[:, 1], c='green', marker='s', label='Class 1')
    plt.scatter(X_class3[:, 0], X_class3[:, 1], c='blue', marker='^', label='Class 2')
    plt.scatter(X_class4[:, 0], X_class4[:, 1], c='black', marker='*', label='Class 3')
    plt.scatter(X_class5[:, 0], X_class5[:, 1], c='black', marker='h', label='Class 4')
    plt.plot(M[:,0], M[:,1], 'o', ms = 10)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best')
    plt.title(loop)
    plt.show()
    

def K_Mean(X, Y, N, K):
    M = init_centroids(X, K)
    loop = 1
    while(True):
        tempM = M
        Y = assign_cluster(X, Y, M, N, K)
        M = assign_centroids(X, K, N)
        if np.array_equal(M,tempM) : break
        else :
            print(loop)
            loop+=1
    #plt.show()
    showResult(X, Y, M, loop)
    


K_Mean(X, Y, N, K)



