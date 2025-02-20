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
    def forward(self, inputs, target): #inputs represents the output of the previous layer and the target is the label of the dataset
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


x = np.random.randn(10, 27*27)
l1 = Layer_Dense(27*27, 200)
a1 = Activation_ReLU()
l2 = Layer_Dense(200, 10)
a2 = Activation_softmax()
loss = Loss_MSE()


l1.forward(x)
a1.forward(l1.output)
l2.forward(a1.output)
a2.forward(l2.output)
target = graph(np.random.randint(1, 11, size = (10,1)))
loss.forward(a2.output, target)
loss.backward()
print(l1.weights.grad[0][0], l1.weights.value[0][0])
'''sub = target - a2.output
sqr = sub**2
sum = sqr.sum()
div = sum / 4
div.backward()
print(l1.weights.grad.shape, l1.weights.value.shape)'''
