import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, -1)
train_labels = train_labels.reshape(60000, -1)
test_images = test_images.reshape(10000, -1)/255
test_labels = test_labels.reshape(10000, -1)
N = 5000
Shape = 28*28
K_init = 10

# X = np.concatenate(train_images, axis=0)
X = test_images
X = X[:N]

random_indices = np.random.choice(N, K_init, replace=False)
M = X[random_indices]
Y = np.zeros((N, K_init))


def init_centroids(Shape, K_init):
    # return X[np.random.choice(len(X), K_init, replace = False)]
    M = np.zeros((K_init, Shape))
    for i in range(10):
        j = 0
        while True:
            if test_labels[j][0] == i : 
                # print(test_labels[j])
                M[i] = test_images[j]
                break
            else : j += 1
    return M


def distance(a, b):
    res = 0
    # print(len(a),len(b))
    # print(a)
    # print(b)
    for i in range(len(a)):
        res += (a[i] - b[i])**2
    return res

def assign_cluster(X, Y, M, N, K_init):
    for i in range(N):
        minDis = 10e10
        for j in range(K_init):
            Y[i][j] = 0
            if distance(X[i], M[j]) < minDis:
                res = j
                minDis = distance(X[i], M[j])
        Y[i][res] = 1
    return Y

def assign_centroids(X, K_init, N, Shape):
    newM = np.zeros((K_init, Shape))
    for i in range(K_init):
        sum = 0
        for j in range(N):
            if Y[j][i] == 1:
                newM[i] += X[j]
                sum += 1
        newM[i] /= sum
    return newM

def showResult(Y, N, K_init):
    dict = {}
    for i in range(N):
        for j in range(K_init):
            if Y[i][j] == 1:
                if test_labels[i][0] not in dict:
                    dict[test_labels[i][0]] = [j]
                else : dict[test_labels[i][0]].append(j)
    print()
    for i in dict:
        x = {}
        for j in dict[i]:
            if j not in x : x[j] = 1
            else : x[j] += 1
        max = 0
        index = 0
        for j in x:
            if x[j] >= max :
                max = x[j]
                index = j
        print(index,max,len(dict[i]),":\n",dict[i])
    

def K_Mean(X, Y, N, K_init, Shape):
    M = init_centroids(Shape, K_init)
    loop = 1
    while(True):
        sys.stdout.write("\rIteration: {}".format(loop,))
        tempM = M
        Y = assign_cluster(X, Y, M, N, K_init)
        M = assign_centroids(X, K_init, N, Shape)
        if np.array_equal(M,tempM) : 
            loop+=1
            break
        else :
            loop+=1
    #plt.show()
    showResult(Y, N, K_init)
    


K_Mean(X, Y, N, K_init, Shape)