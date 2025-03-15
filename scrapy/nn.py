import numpy as np
import pickle as pkl
from autograd import graph
import idx2numpy
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
#=nb



class Layer_Dense:
    def __init__(self, n_inputs : int, n_neurons : int):

        self.weights = graph((0.1 * np.random.randn(n_inputs, n_neurons)))
        self.biases = graph(np.random.randn(1, n_neurons))
    
    def forward(self, inputs):
    
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        mul = (self.inputs * self.weights) 
        self.output = mul + self.biases
        return self.output
    
    def backward(self):
        return self.output.backward()
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.output =  self.inputs.ReLU()
        return self.output
    def backward(self):
        return self.output.backward()            
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.output = 1 / (1 + (inputs * -1).e())
        return self.output
    def backward(self):
        return self.output.backward()
class Activation_softmax:
    def forward(self, inputs):
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.output = self.inputs.softmax()
    def backward(self):
        return self.output.backward()        
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.output = inputs
        return self.output
    def backward(self):
        return self.output.backward()
class Loss_MSE:
    def forward(self, inputs, target): #inputs represents the output of the previous layer (prediction) and the target is the label of the dataset
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.target = target if isinstance(target, graph) else graph(target)
        sub = self.target - self.inputs
        sqr = sub**2
        sum = sqr.sum()
        div = sum /self.target.value.shape[1]
        self.output = div
        return self.output
    def backward(self):
        return self.output.backward()
class Loss_Catagorical:
    def forward(self, inputs, targets):
        assert isinstance(inputs, graph) 
        assert len(np.array(targets).shape) < 2 # no one hot encoding
        clipped = inputs.clip(1e-7, 1 - 1e-7)
        clip_index = clipped[range(len(targets)), targets]
        self.output = -(clip_index.log().sum()/len(targets))  
        return self.output
    def backward(self):
        return self.output.backward()
class Optimizer_Adam:
    def __init__(self):
        self.iterations = 0


#img,target = idx2numpy.convert_from_file("C:\\Users\\pro\\Documents\\GitHub\\scrapy\\dataset\\train-images.idx3-ubyte"), idx2numpy.convert_from_file("C:\\Users\\pro\\Documents\\GitHub\\scrapy\\dataset\\train-labels.idx1-ubyte")
X, y = spiral_data(samples=100, classes=3)

l1 = Layer_Dense(2, 123)
a1 = Activation_ReLU()
l2 = Layer_Dense(123, 3)
a2 = Activation_softmax()
loss = Loss_Catagorical()
for i in range(10):
    print(f'EPOCH ----------------------------- {i}')
    l1.forward(X.reshape(-1, 2))
    a1.forward(l1.output)
    l2.forward(a1.output)
    a2.forward(l2.output)
    loss.forward(a2.output, y)
    print(loss.output)
    loss.backward()
    l1.weights -= l1.weights.grad * 0.01
    l1.biases -= l1.biases.grad * 0.01
    l2.weights -= l1.weights.grad * 0.01
    l2.biases -= l2.biases.grad * 0.01
    





