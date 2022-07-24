
import os

import flwr as fl
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pickle as pckl
import numpy as np



def load_data(client_id:int, num_of_clients:int):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / num_of_clients)
    idx_from, idx_to = client_id * partition_size, (client_id + 1) * partition_size
    x_cid = x_train[idx_from:idx_to] / 255.0
    y_cid = y_train[idx_from:idx_to]

    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(x_cid) * 0.9)
    x_train_cid, y_train_cid = x_cid[:split_idx], y_cid[:split_idx]
    x_val_cid, y_val_cid = x_cid[split_idx:], y_cid[split_idx:]

    return x_train_cid, y_train_cid, x_val_cid, y_val_cid 

x_train, y_train, x_test, y_test = load_data(0,4)

count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0

for i in range(13500):
    if(y_train[i]==0):
        count0=count0+1
    if(y_train[i]==1):
        count1=count1+1
    if(y_train[i]==2):
        count2=count2+1
    if(y_train[i]==3):
        count3=count3+1
    if(y_train[i]==4):
        count4=count4+1
    if(y_train[i]==5):
        count5=count5+1
    if(y_train[i]==6):
        count6=count6+1
    if(y_train[i]==7):
        count7=count7+1
    if(y_train[i]==8):
        count8=count8+1
    if(y_train[i]==9):
        count9=count9+1

print("Client0")
print(count0)
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
print(count6)
print(count7)
print(count8)
print(count9)

# creating the dataset
data = {'0':count0, '1':count1, '2':count2, '3':count3, '4':count4, '5':count5, '6':count6,'7':count7, '8':count8, '9':count9}
courses = list(data.keys())
values = list(data.values())

plt.subplot(2, 2, 1)
plt.bar(courses[0], values[0], color ='r',width = 1)
plt.bar(courses[1], values[1], color ='b',width = 1)
plt.bar(courses[2], values[2], color ='g',width = 1)
plt.bar(courses[3], values[3], color ='grey',width = 1)
plt.bar(courses[4], values[4], color ='yellow',width = 1)
plt.bar(courses[5], values[5], color ='maroon',width = 1)
plt.bar(courses[6], values[6], color ='purple',width = 1)
plt.bar(courses[7], values[7], color ='orange',width = 1)
plt.bar(courses[8], values[8], color ='cyan',width = 1)
plt.bar(courses[9], values[9], color ='pink',width = 1)
plt.title("Client0")
#plt.grid(color = "grey", linewidth = "1", linestyle = "-")
#plt.grid(axis='y', color = 'black', linestyle = '--', linewidth = 0.5)

for index, value in enumerate(courses):
    plt.text(value, index,
             str(values[index]), fontweight = 'bold', fontsize = 12,verticalalignment='bottom',horizontalalignment='center')


x_train, y_train, x_test, y_test = load_data(1,4)

count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0

for i in range(13500):
    if(y_train[i]==0):
        count0=count0+1
    if(y_train[i]==1):
        count1=count1+1
    if(y_train[i]==2):
        count2=count2+1
    if(y_train[i]==3):
        count3=count3+1
    if(y_train[i]==4):
        count4=count4+1
    if(y_train[i]==5):
        count5=count5+1
    if(y_train[i]==6):
        count6=count6+1
    if(y_train[i]==7):
        count7=count7+1
    if(y_train[i]==8):
        count8=count8+1
    if(y_train[i]==9):
        count9=count9+1

print("Client1")
print(count0)
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
print(count6)
print(count7)
print(count8)
print(count9)

data = {'0':count0, '1':count1, '2':count2, '3':count3, '4':count4, '5':count5, '6':count6,'7':count7, '8':count8, '9':count9}
courses = list(data.keys())
values = list(data.values())

plt.subplot(2, 2, 2)
plt.bar(courses[0], values[0], color ='r',width = 1)
plt.bar(courses[1], values[1], color ='b',width = 1)
plt.bar(courses[2], values[2], color ='g',width = 1)
plt.bar(courses[3], values[3], color ='grey',width = 1)
plt.bar(courses[4], values[4], color ='yellow',width = 1)
plt.bar(courses[5], values[5], color ='maroon',width = 1)
plt.bar(courses[6], values[6], color ='purple',width = 1)
plt.bar(courses[7], values[7], color ='orange',width = 1)
plt.bar(courses[8], values[8], color ='cyan',width = 1)
plt.bar(courses[9], values[9], color ='pink',width = 1)
plt.title("Client1")
#plt.grid(color = "grey", linewidth = "1", linestyle = "-")
#plt.grid(axis='y', color = 'black', linestyle = '--', linewidth = 0.5)

for index, value in enumerate(courses):
    plt.text(value, index,
             str(values[index]), fontweight = 'bold', fontsize = 12,verticalalignment='bottom',horizontalalignment='center')


x_train, y_train, x_test, y_test = load_data(2,4)

count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0

for i in range(13500):
    if(y_train[i]==0):
        count0=count0+1
    if(y_train[i]==1):
        count1=count1+1
    if(y_train[i]==2):
        count2=count2+1
    if(y_train[i]==3):
        count3=count3+1
    if(y_train[i]==4):
        count4=count4+1
    if(y_train[i]==5):
        count5=count5+1
    if(y_train[i]==6):
        count6=count6+1
    if(y_train[i]==7):
        count7=count7+1
    if(y_train[i]==8):
        count8=count8+1
    if(y_train[i]==9):
        count9=count9+1

print("Client2")
print(count0)
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
print(count6)
print(count7)
print(count8)
print(count9)

data = {'0':count0, '1':count1, '2':count2, '3':count3, '4':count4, '5':count5, '6':count6,'7':count7, '8':count8, '9':count9}
courses = list(data.keys())
values = list(data.values())

plt.subplot(2, 2, 3)
plt.bar(courses[0], values[0], color ='r',width = 1)
plt.bar(courses[1], values[1], color ='b',width = 1)
plt.bar(courses[2], values[2], color ='g',width = 1)
plt.bar(courses[3], values[3], color ='grey',width = 1)
plt.bar(courses[4], values[4], color ='yellow',width = 1)
plt.bar(courses[5], values[5], color ='maroon',width = 1)
plt.bar(courses[6], values[6], color ='purple',width = 1)
plt.bar(courses[7], values[7], color ='orange',width = 1)
plt.bar(courses[8], values[8], color ='cyan',width = 1)
plt.bar(courses[9], values[9], color ='pink',width = 1)
plt.title("Client2")
#plt.grid(color = "grey", linewidth = "1", linestyle = "-")
#plt.grid(axis='y', color = 'black', linestyle = '--', linewidth = 0.5)

for index, value in enumerate(courses):
    plt.text(value, index,
             str(values[index]), fontweight = 'bold', fontsize = 12,verticalalignment='bottom',horizontalalignment='center')

x_train, y_train, x_test, y_test = load_data(3,4)

count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0

for i in range(13500):
    if(y_train[i]==0):
        count0=count0+1
    if(y_train[i]==1):
        count1=count1+1
    if(y_train[i]==2):
        count2=count2+1
    if(y_train[i]==3):
        count3=count3+1
    if(y_train[i]==4):
        count4=count4+1
    if(y_train[i]==5):
        count5=count5+1
    if(y_train[i]==6):
        count6=count6+1
    if(y_train[i]==7):
        count7=count7+1
    if(y_train[i]==8):
        count8=count8+1
    if(y_train[i]==9):
        count9=count9+1

print("Client3")
print(count0)
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
print(count6)
print(count7)
print(count8)
print(count9)

data = {'0':count0, '1':count1, '2':count2, '3':count3, '4':count4, '5':count5, '6':count6,'7':count7, '8':count8, '9':count9}
courses = list(data.keys())
values = list(data.values())

plt.subplot(2, 2, 4)
plt.bar(courses[0], values[0], color ='r',width = 1)
plt.bar(courses[1], values[1], color ='b',width = 1)
plt.bar(courses[2], values[2], color ='g',width = 1)
plt.bar(courses[3], values[3], color ='grey',width = 1)
plt.bar(courses[4], values[4], color ='yellow',width = 1)
plt.bar(courses[5], values[5], color ='maroon',width = 1)
plt.bar(courses[6], values[6], color ='purple',width = 1)
plt.bar(courses[7], values[7], color ='orange',width = 1)
plt.bar(courses[8], values[8], color ='cyan',width = 1)
plt.bar(courses[9], values[9], color ='pink',width = 1)
plt.title("Client3")
#plt.grid(color = "grey", linewidth = "1", linestyle = "-")
#plt.grid(axis='y', color = 'black', linestyle = '--', linewidth = 0.5)

for index, value in enumerate(courses):
    plt.text(value, index,
             str(values[index]), fontweight = 'bold', fontsize = 12,verticalalignment='bottom',horizontalalignment='center')

plt.show()