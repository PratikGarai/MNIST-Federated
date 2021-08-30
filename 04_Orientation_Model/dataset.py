import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
plt.rcParams["figure.figsize"] = (10,10)


# Print a grid 
def printGrid(data, m, n) :
    for ind, i in enumerate(data):
        ax = plt.subplot(m, n, ind+1)
        plt.imshow(i, cmap="gray")
    plt.show()

def printGridSlice(data, m, n) :
    l = (m*n)
    for i in range(l) :
        ax = plt.subplot(m, n, i+1)
        plt.imshow(data[i], cmap="gray")
    plt.show()


# FIlter the dataset and limit the results
def filterAndLimit(target, limit, datax , datay) :
    count = 0
    i = 0
    x = []
    y = []
    if limit==-1 :
        limit = len(datax)
    while(count<limit and i<limit) :
        if target==datay[i] :
            x.append(datax[i])
            y.append(datay[i])
            count += 1
        i += 1
    
    return x,y,count


# Take data and generated the rotated dataset
def getRandomOrientationData(data, counts = None) :
    res = []
    y = []
    l = len(data)
    if not counts : 
        counts = []
        for i in range(4) :
            counts.append(l//4)
    
    assert sum(counts) <= l
            
    for im in data:
        orientation = random.randint(0, 3)
        res.append(np.rot90(im, orientation, (1, 0)))
        y.append(orientation)
    
    return res, y, counts


# Driver pipleline to handle the dataset generation
def getDataset(datax, datay, target, limit=-1, counts=None, plotDemo=None) :
    x,y,count = filterAndLimit(target, limit, datax, datay)
    x,y,counts = getRandomOrientationData(x, counts)
    if plotDemo : 
        printGridSlice(x, *plotDemo)
    return x,y,counts


# Main Dataset class
class Dataset : 
    def __init__(self, x = [], y = [], counts = [0, 0, 0, 0]) :
        self.x = x
        self.y = y
        self.length = len(x)
        self.distribution = counts
    
    def save(self, filename) :
        with open(f'{filename}.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, filename) :
        with open(f'{filename}.pickle', 'rb') as handle:
            b = pickle.load(handle)
            self.x = b.x
            self.y = b.y
            self.length = len(b.x)
            self.distribution = b.distribution
    
    def printSlice(self, m, n) :
        assert m*n <= self.length
        for i in range(m*n) :
            ax = plt.subplot(m, n, i+1)
            plt.imshow(self.x[i], cmap="gray")
        plt.show()
        print(self.y[:m*n])
    
    def plotDist(self) :
        plt.hist(self.y)
        plt.show()


# Handler for MNIST
import tensorflow as tf

class MNISTDataset(Dataset) :
    '''
        Generates and MNIST Dataset distibution with the given target's
        images of varied orientation. Also uses a four element list counts[]
        to control the count of images of given orientation
    '''
    def __init__(self, target = 0, counts = None) :
        (x_train, y_train) , _ = tf.keras.datasets.mnist.load_data()
        x, y, counts = getDataset(x_train, y_train, target)
        super(MNISTDataset, self).__init__(x, y, counts)
