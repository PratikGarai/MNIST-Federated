import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
