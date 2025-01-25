import numpy as np
import pickle as pkl
from autograd import graph

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
    pass        
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs if isinstance(inputs, graph) else graph(inputs)
        self.output = inputs
        return self.output
    def backward(self):
        return self.output.backward()

x = np.random.randn(10, 27*27)
l1 = Layer_Dense(27*27, 200)
a1 = Activation_ReLU()
l2 = Layer_Dense(200, 10)


l1.forward(x)
a1.forward(l1.output)
l2.forward(a1.output)


softmax = l2.output.value
print(softmax[0])
