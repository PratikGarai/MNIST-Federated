import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# AUxillary methods
# def getDist(y):
#     ax = sns.countplot(y)
#     ax.set(title="Count of data classes")
#     plt.show()

# def getData(dist, x, y):
#     dx = []
#     dy = []
#     counts = [0 for i in range(10)]
#     for i in range(len(x)):
#         if counts[y[i]]<dist[y[i]]:
#             dx.append(x[i])
#             dy.append(y[i])
#             counts[y[i]] += 1
        
#     return np.array(dx), np.array(dy)

# # Load and compile Keras model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# x_train, y_train = getData(dist, x_train, y_train)
# getDist(y_train)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self,config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
#         hist = r.history
#         print("Fit history : " ,hist)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
#         print("Eval accuracy : ", accuracy)
#         return loss, len(x_test), {"accuracy": accuracy}

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
    

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
