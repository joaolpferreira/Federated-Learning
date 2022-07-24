import dataloader
from matplotlib import pyplot as plt

x_train_1, y_train_1, x_val_1, y_val_1 = dataloader.load_data(0,4)
x_train_2, y_train_2, x_val_2, y_val_2  = dataloader.load_data(1,4)
x_train_3, y_train_3, x_val_3, y_val_3 = dataloader.load_data(2,4)
x_train_4, y_train_4, x_val_4, y_val_4  = dataloader.load_data(3,4)



fig = plt.figure()
for i in range(200):
    if(i<50):
        plt.subplot(23,10,i+1)
        plt.imshow(x_train_1[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        if(i==20):
            plt.ylabel("Client 0",fontsize=18)
            
    elif(i<100):
        plt.subplot(23,10,i+11)
        plt.imshow(x_train_2[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        if(i==70):
            plt.ylabel("Client 1",fontsize=18)
    elif(i<150):
        plt.subplot(23,10,i+21)
        plt.imshow(x_train_3[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        if(i==120):
            plt.ylabel("Client 2",fontsize=18)
    else:
        plt.subplot(23,10,i+31)
        plt.imshow(x_train_4[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        if(i==170):
            plt.ylabel("Client 3",fontsize=18)

_=plt.show()

