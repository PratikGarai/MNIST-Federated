import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import MNISTDataset, printGridSlice
import sys

# Load and compile Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=164, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='softmax')
])
model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)

# Load dataset
datas = [4, 5]
print("This client has data of  : ", datas)
dataset = MNISTDataset(datas)
x_train, x_test, y_train, y_test = train_test_split(np.array(dataset.x), np.array(dataset.y), test_size = 0.15)
x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
