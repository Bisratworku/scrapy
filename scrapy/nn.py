import numpy as np
from autograd import graph
import idx2numpy
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
    def __init__(self,learning_rate = 1., decay = 0.0, momentum = 0.9):#in this case momentem is bata an beta is usually 0.9 beta is used to control the influence of the previous gradient on the current one
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
        corrected_momentum_weights = layer.momentum_weights / (1 + (self.beta_1**(layer.iter))) # bias correction 
        layer.momentum_biases = self.beta_1 * layer.momentum_biases + (1 - self.beta_1) * layer.biases.grad
        corrected_momentum_biases = layer.momentum_biases / (1 + (self.beta_1**(layer.iter))) # bias correction 
        layer.squared_weights = self.beta_2 * layer.squared_weights + (1 - self.beta_2) * layer.weights.grad**2
        corrected_squared_weights = layer.squared_weights / (1 + (self.beta_2**(layer.iter)))
        layer.squared_biases = self.beta_2 * layer.squared_biases + (1 - self.beta_2) * layer.biases.grad**2
        corrected_squared_biases =  layer.squared_biases / (1 + (self.beta_2**(layer.iter)))
        layer.iter += 1
        layer.weights -= (corrected_momentum_weights / (self.epsilon + np.sqrt(corrected_squared_weights))) * self.update()  
        layer.biases -=  (corrected_momentum_biases / (self.epsilon + np.sqrt(corrected_squared_biases))) * self.update() 

class model:
    def __init__(self, layers , loss, optimaizer):
        self.layers = layers
        self.loss = loss
        self.optimaizer = optimaizer
    def train(self, train_data, train_label):
        tunnable = []
        self.data = train_data
        for i in self.layers:
            i.forward(self.data)
            if isinstance(i, Layer_Dense):
                tunnable.append(i)
            self.data = i.output
        self.loss.forward(self.data, train_label)
        self.loss.backward()
        for j in reversed(tunnable):
            self.optimaizer.step(j)
    def test(self, test_data, test_label):
        self.test_data = test_data
        for i in self.layers:
            i.forward(self.test_data)
            self.test_data = i.output
        
train_img = idx2numpy.convert_from_file('C:/Users/Bisrat/Documents/GitHub/scrapy/train-images.idx3-ubyte')/255
train_lbl = idx2numpy.convert_from_file('C:/Users/Bisrat/Documents/GitHub/scrapy/train-labels.idx1-ubyte')
test_img = idx2numpy.convert_from_file('C:/Users/Bisrat/Documents/GitHub/scrapy/t10k-images.idx3-ubyte')/255
test_lbl = idx2numpy.convert_from_file('C:/Users/Bisrat/Documents/GitHub/scrapy/t10k-labels.idx1-ubyte')
batch = 32
train_data =  list(zip(np.array_split(train_img.reshape(-1,28*28),batch), np.array_split(train_lbl, batch)))
test_data = list(zip(np.array_split(test_img.reshape(-1,28*28), batch), np.array_split(test_lbl, batch)))


'''
for i in range(5):
    print(f'The ------------ {i} ---------------- iteration')
    for batch, (x,y) in enumerate(train_data):
        print(f'_________________________ {batch + 1} ___________________________')
        l1.forward(x)
        a1.forward(l1.output)
        l2.forward(a1.output)
        a2.forward(l2.output)
        l3.forward(a2.output)
        a3.forward(l3.output)
        loss.forward(a3.output,y)
        print(loss.output.value)
        loss.backward()
        optimizer.step(l3)
        optimizer.step(l2)
        optimizer.step(l1)
'''

x,y = train_data[0]
m = model([Layer_Dense(28*28, 200),
           Activation_ReLU(),
           Layer_Dense(200, 100),
           Activation_ReLU(),
           Layer_Dense(100, 10),
           Activation_softmax()], Loss_Catagorical(), Optimizer_ADAM(learning_rate=0.001, decay=1e-3))
for i in range(5):
    print(f'The ------------ {i} ---------------- iteration')
    for batch, (x,y) in enumerate(train_data):
        print(f'_________________________ {batch + 1} ___________________________')
        m.train(x,y)
