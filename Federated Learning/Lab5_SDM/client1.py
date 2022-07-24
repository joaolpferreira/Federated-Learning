import os

import flwr as fl
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pickle as pckl
import numpy as np
# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),        
    ]
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
x_train, y_train, x_test, y_test = load_data(0,4)



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        print(model.get_weights())
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=450)
     
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("localhost:8080", client=CifarClient())



