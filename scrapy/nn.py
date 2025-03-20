import numpy as np
import pickle as pkl
from autograd import graph
import idx2numpy
import matplotlib.pyplot as plt
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
class Optimizer_SGD:
    def __init__(self,learning_rate, decay, momentum = 0.9):#in this case momentem is bata an beta is usually 0.9 beta is used to control the influence of the previous gradient on the current one
        self.learning_rate = learning_rate
        self.decay = decay
        self.epoch = 0
        self.current_lr = 0
        self.momentum = momentum
    def update(self):
        self.epoch += 1 
        self.current_lr = (1/(1+self.decay * self.epoch)) * self.learning_rate
        return self.current_lr
    def step(self, layer):
        if not hasattr(layer, "momentum_weights"):
            layer.momentum_weights = np.zeros_like(layer.weights.value)
            layer.momentum_biases =  np.zeros_like(layer.biases.value)
        layer.momentum_weights = self.momentum * layer.momentum_weights + (1 - self.momentum) * layer.weights.grad #momentum is used to prevent the model from getting stuck in local minima to for further eplanaton click on the link https://www.youtube.com/watch?v=k8fTYJPd3_I
        layer.momentum_biases = self.momentum * layer.momentum_biases + (1 - self.momentum) * layer.biases.grad
        layer.weights -= layer.momentum_weights * self.update()
        layer.biases -=  layer.momentum_biases * self.update() 
class Optimizer_RMSprop:
    def __init__(self, learning_rate, decay = 0.0 , momentum = 1e-7, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_lr = 0
        self.epoch = 0
        self.momentum = momentum
        self.epsilon = epsilon
    def update(self):
        self.epoch += 1 
        self.current_lr = (1/(1+self.decay * self.epoch)) * self.learning_rate
        return self.current_lr
    def step(self, layer):
        if not hasattr(layer, "momentum_weights"):
            layer.momentum_weights = np.zeros_like(layer.weights.value)
            layer.momentum_biases =  np.zeros_like(layer.biases.value)
        layer.momentum_weights = self.momentum * layer.momentum_weights + (1 - self.momentum) * layer.weights.grad**2 
        layer.momentum_biases = self.momentum * layer.momentum_biases + (1 - self.momentum) * layer.biases.grad**2
        layer.weights -=  (layer.weights.grad / (self.epsilon + np.sqrt(layer.momentum_weights))) * self.update()
        layer.biases -=  (layer.biases.grad / (self.epsilon + np.sqrt(layer.momentum_biases))) * self.update()
class Optimizer_ADAM: # in adam optimization we are combining momenentum with RMS prop 
    def __init__(self, learning_rate = 1e-3 , decay = 0.0 ,beta_1 = 0.9 , beta_2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.epoch = 0
    def update(self):
        self.epoch += 1 
        self.current_lr = (1/(1+self.decay * self.epoch)) * self.learning_rate
        return self.current_lr    
    def step(self, layer):
        if not hasattr(layer, "momentum_weights"):
            layer.momentum_weights = np.zeros_like(layer.weights.value) # both are for the momentum part 
            layer.momentum_biases = np.zeros_like(layer.biases.value)
            layer.iter = 1
            layer.squared_weights = np.zeros_like(layer.weights.value) # bothe are for RMS part
            layer.squared_biases = np.zeros_like(layer.biases.value)   
        layer.momentum_weights = self.beta_1 * layer.momentum_weights + (1 - self.beta_1) * layer.weights.grad
        corrected_momentum_weights = layer.momentum_weights / (1 + self.beta_1**(layer.iter)) # bias correction 
        layer.momentum_biases = self.beta_1 * layer.momentum_biases + (1 - self.beta_1) * layer.biases.grad
        corrected_momentum_biases = layer.momentum_biases / (1 + self.beta_1**(layer.iter)) # bias correction 
        layer.squared_weights = self.beta_2 * layer.squared_weights + (1 - self.beta_2) * layer.weights.grad**2
        corrected_squared_weights = layer.squared_weights / (1 + self.beta_2**(layer.iter))
        layer.squared_biases = self.beta_2 * layer.squared_biases + (1 - self.beta_2) * layer.biases.grad**2
        corrected_squared_biases =  layer.squared_biases / (1 + self.beta_2**(layer.iter))
        layer.iter += 1
        layer.weights -= self.update() * (corrected_momentum_weights / (self.epsilon + np.sqrt(corrected_squared_weights)))  
        layer.biases -= self.update() * (corrected_momentum_biases / (self.epsilon + np.sqrt(corrected_squared_biases))) 
        


#img,target = idx2numpy.convert_from_file("C:\\Users\\pro\\Documents\\GitHub\\scrapy\\dataset\\train-images.idx3-ubyte"), idx2numpy.convert_from_file("C:\\Users\\pro\\Documents\\GitHub\\scrapy\\dataset\\train-labels.idx1-ubyte")

X, y = spiral_data(samples=100, classes=3)

l1 = Layer_Dense(2, 200)
a1 = Activation_ReLU()
l2 = Layer_Dense(200,3)
a2 = Activation_softmax()
loss = Loss_Catagorical()
optim = Optimizer_ADAM()

for i in range(10):
    l1.forward(X.reshape(-1,2))
    a1.forward(l1.output)
    l2.forward(a1.output)
    a2.forward(l2.output)
    loss.forward(a2.output,y)
    print(loss.output.value)
    optim.step(l2)
    optim.step(l1)