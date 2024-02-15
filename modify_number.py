import tensorflow as tf
import numpy as np
import sys
import cv2 
# Tải bộ dữ liệu MNIST
mnist = tf.keras.datasets.mnist


# Chia thành dữ liệu huấn luyện và kiểm tra
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of testing labels:", test_labels.shape)
train_images = train_images.reshape(60000, -1)/255
train_labels = train_labels.reshape(60000, -1)
test_images = test_images.reshape(10000, -1)/255
test_labels = test_labels.reshape(10000, -1)

def SM(z):
    z=(z+1000)/500
    ez=np.exp(z)
    res=0
    for i in ez[0]:
        res=res+i
    res=[[res]]
    for i in range(1,len(ez)):
        res=np.vstack((res,[sum(ez[i])]))
    return ez/res

def pred(x,w,bias):
    z=x@w+bias
    return SM(z)

def Convert(x):
    cell=[]
    for i in range(28):
        for j in range(28):
            cell.append(x[i][j])
    return cell

def changeLabel(b):

    # Kích thước của ma trận đầu ra
    m = b.shape[0]
    n = 10  # Số cột trong ma trận đầu ra

    # Tạo ma trận đầu ra với tất cả giá trị là 0
    result = np.zeros((m, n))

    # Đánh dấu vị trí tương ứng với giá trị trong vector b là 1
    result[np.arange(m), b.reshape(-1)] = 1

    # In ra ma trận kết quả
    return result
train_labels__test = train_labels
train_labels = changeLabel(train_labels)
file_name_bias = "bias_matrix - Copy.txt"
file_name_w = "w_matrix - Copy.txt"
# w = np.full((28*28, 10), 0.1)
# bias = np.full(10, 0.34)
w = np.loadtxt(file_name_w)
bias = np.loadtxt(file_name_bias)
lr=0.15
iner = 100
# for i in range(iner):
#     z=train_images@w+bias
#     pre=SM(z)

#     w=w-lr*(train_images.T@(pre-train_labels))
#     bias=bias-lr*(sum(pre-train_labels))

#     sys.stdout.write("\rIteration: {}/{}".format(i + 1, iner))


np.savetxt(file_name_w, w)
np.savetxt(file_name_bias, bias)



cost=0
z=train_images@w+bias # 500 x 400 vs 400 x 2 = 500 x 2
pre=SM(z)
for m in range(len(train_labels)):
    for n in range(10):
        cost-=train_labels[m][n]*np.log(pre[m][n])
print("\nCost : ",cost/len(train_images))

total = 0
dic={}
# print(SM([test_images[3]]@w+bias))
for i in range(len(test_labels)):
    if SM([test_images[i]]@w+bias)[0][test_labels[i]] > 0.5 : 
        total+=1
    else : 
        if str(i//250) not in dic : dic[str(i//250)]=1
        else : dic[str(i//250)]+=1
print("Test data :"+str(total)+"/"+str(len(test_labels)))

total = 0
dic={}
# print(SM([test_images[3]]@w+bias))
for i in range(len(train_labels__test)):
    if SM([train_images[i]]@w+bias)[0][train_labels__test[i]] > 0.5 : 
        total+=1
    else : 
        if str(i//250) not in dic : dic[str(i//250)]=1
        else : dic[str(i//250)]+=1
print("Training data :"+str(total)+"/"+str(len(train_labels__test)))

img=cv2.imread('image_test/9_2.png',0)
test=Convert(img/255)
result = SM([test]@w+bias)[0]*100
probability = 0
for i in range(10) :
    if result[i] > probability :
        probability = result[i]
        index = i
print("Result :",index)
print("Probability : ",round(probability,2))