from typing import Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import flwr as fl
import tensorflow as tf
import os
import keras
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    


'''
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


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_train, y_train, x_val, y_val  = load_data(3, 3)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, accuracy

    return evaluate

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),        
    ]
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

evaluate = get_eval_fn(model)
print(evaluate[0])

plt.plot(evaluate[0])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''

strategy = fl.server.strategy.FedAvg(
    #fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=4,  # Minimum number of clients to be sampled for the next round
    min_available_clients=4,  # Minimum number of clients that need to be connected to the server before a training round can start
)

a=fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 6},
    force_final_distributed_eval = True,
    strategy=strategy,
    )
import re
print(a)


aux=re.split(": |\n",str(a))


import numpy as np
lossy=[aux[2], aux[4], aux[6], aux[8], aux[10], aux[12]]



#lossy.sort()
#print("Array1 " + str(lossy))
#lossy[::-1].sort()
#print("Array2 " + str(lossy))
lossy=np.array(lossy)

y = lossy.astype(np.float)
print(y) # Output : [1.1, 2.2, 3.3]


x_ax=np.array([1,2,3,4,5,6])

#print(int(aux[0])+int(aux[1]))
#print(int(aux[0]))
#print(lossy[0]+lossy[1])
print("Array " + str(lossy))
#lossy[::-1].sort()
plt.plot(x_ax,y)
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
#plt.gca().invert_yaxis()
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
