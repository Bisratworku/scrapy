import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import idx2numpy
#=nb
train_img =  idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_lbl = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
test_img = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
test_lbl = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

class Layer_Dense:
    def __init__(self, n_inputs : int, n_neurons : int,
                 weight_regularizer_l1 : float = 0, weight_regularizer_l2 :float = 0,
                 bias_regularizer_l1 :float = 0, bias_regularizer_l2 :float =0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)
l1 = Layer_Dense(28*28, 123)
img = train_img[:10].reshape(-1, 28 * 28)
l1.forward(img)
class conv2D:
    def __init__(self, * , kernel : tuple, padding : int = 0, stride :int = 1):
        self.weights = np.random.randn(*kernel)
        self.padding = padding
        self.stride = stride
        self.kernel = kernel
        self.deapth = kernel[0] 
    def zero_pad(self, img):
        if len(img.shape) == 2:
            return np.pad(img, self.padding, "constant")
        if len(img.shape) == 3:
            return np.pad(img.sum(-1)/255 , self.padding, "constant")
        else:
            return "Please insert a proper image"
    def convolve(self, img, kernel):
        pad_img = self.zero_pad(img)
        output = np.zeros((self.output.shape[2], self.output.shape[3]))
        for idx, i in enumerate(range(0,pad_img.shape[0] - (kernel.shape[0] - 1), self.stride)):
            for idj,j in enumerate(range(0, pad_img.shape[1] - (kernel.shape[1] - 1), self.stride)):
                 output[idx][idj] = np.sum(pad_img[i: i + kernel.shape[0] , j : j + kernel.shape[1]] * kernel)
        return output.clip(0, 255)
    def forward(self, img):
        try:
            if len(img.shape) == 3:
                x_out = int(np.ceil((((img.shape[1] - self.kernel[1]) + (2 * self.padding))/self.stride) + 1))
                y_out = int(np.ceil((((img.shape[2] - self.kernel[2]) + (2 * self.padding))/self.stride) + 1))
            elif len(img.shape) == 4:
                x_out = int(np.ceil((((img.shape[2] - self.kernel[1]) + (2 * self.padding))/self.stride) + 1))
                y_out = int(np.ceil((((img.shape[3] - self.kernel[2]) + (2 * self.padding))/self.stride) + 1))
            self.output = np.zeros((self.deapth,img.shape[0],x_out, y_out))
            self.biases = np.zeros((self.deapth,img.shape[0] , 1, 1))
            for i in range(self.deapth):
                for idx,j in enumerate(img):
                    self.output[i][idx] = self.convolve(j,self.weights[i]) 
            self.output = self.output + self.biases
            return self.output
        except:
            return "the image can no longer be convolved"
img = train_img[:10]
class MaxPool_2D:
    def pool(self, img):
        x_out = int(np.ceil(((img.shape[0] - 2)/2) + 1))
        y_out = int(np.ceil(((img.shape[1] - 2)/2) + 1))
        output = np.zeros((x_out, y_out))
        for idx,i in enumerate(range(0, img.shape[0] - 1, 2)):
            for idj,j in enumerate(range(0, img.shape[1] - 1, 2)):
                output[idx][idj] = np.max(img[i : i + 2, j : j + 2])
        return output
    def forward(self, img):
        try:
            x_out = int(np.ceil(((img.shape[2] - 2)/ 2 ) + 1))
            y_out = int(np.ceil(((img.shape[3] - 2)/2 ) + 1))
            self.output = np.zeros((img.shape[0], img.shape[1], x_out, y_out))
            for idx, i in enumerate(img):
                for idj , j in enumerate(i):
                    self.output[idx][idj] = self.pool(j)
            return self.output
        except:
            return "the image can no longer be downgraded"

class Layer_Dropout:
    def __init__(self, rate : float):
        self.rate = 1 - rate
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
class Optimizer_SGD:
    def __init__(self, learning_rate :float = 1., decay :float = 0., momentum :float = 0.):
        self.learning_rate :float = learning_rate
        self.current_learning_rate :float = learning_rate
        self.decay :float = decay
        self.iterations :int = 0
        self.momentum :float = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adagrad:
    def __init__(self, learning_rate : float = 1., decay :float = 0., epsilon :float =1e-7):
        self.learning_rate :float = learning_rate
        self.current_learning_rate :float = learning_rate
        self.decay :float = decay
        self.iterations :int = 0
        self.epsilon :float = epsilon
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
class Optimizer_RMSprop:
    def __init__(self, learning_rate :float =0.001, decay :float =0., epsilon :float =1e-7,
                 rho :float =0.9):
        self.learning_rate  :float = learning_rate
        self.current_learning_rate :float = learning_rate
        self.decay :float = decay
        self.iterations :int = 0
        self.epsilon :float = epsilon
        self.rho :float = rho
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adam:
    def __init__(self, learning_rate :float =0.001, decay : float =0., epsilon :float =1e-7,
                 beta_1 :float =0.9, beta_2 :float = 0.999):
        self.learning_rate :float = learning_rate
        self.current_learning_rate :float = learning_rate
        self.decay :float = decay
        self.iterations :int = 0
        self.epsilon :float = epsilon
        self.beta_1 :float = beta_1
        self.beta_2 :float = beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums /(1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
class Loss:
    def regularization_loss(self, layer):
        regularization_loss = 0
        l2_weight = []
        l2_bias = []
        l1_weight = []
        l1_bias = []
        for Layer in layer:
            if Layer.weight_regularizer_l1 > 0:
                l1_weight.append( np.sum(np.abs(Layer.weights)))
            if Layer.weight_regularizer_l2 > 0:
                l2_weight.append(np.sum(Layer.weights * Layer.weights))
            if Layer.bias_regularizer_l1 > 0:
                l1_bias.append(np.sum(np.abs(Layer.biases)))
            if Layer.bias_regularizer_l2 > 0:
                l2_bias.append(np.sum(Layer.biases *Layer.biases))
            regularization_loss += np.sum(l2_weight) * Layer.weight_regularizer_l2
            regularization_loss += np.sum(l2_bias) * Layer.bias_regularizer_l2
            regularization_loss += np.sum(l1_weight) * Layer.weight_regularizer_l1
            regularization_loss += np.sum(l1_bias) * Layer.bias_regularizer_l1
        return regularization_loss
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.output =   data_loss
        return data_loss    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum( y_pred_clipped * y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return  negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
    def accuracy(self, pred, target):
        self.prediction = np.argmax(pred, axis = 1)
        if len(target.shape) == 2:
            target = np.argmax(target, axis = 1)
        output = np.mean(self.prediction == target)
        return output
        
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues -(1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
    def accuracy(self, pred, target):
        self.prediction = np.round(pred)
        output = np.mean(self.prediction == target)
        return output
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
    def accuracy(self, y_pred, y_true):
        return  np.mean((y_true - y_pred)**2, axis = -1)
class Loss_MeanAbsoluteError(Loss):  
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
    def accuracy(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis = -1)
class model:
    def __init__(self):
        self.layers = []
        self.tunable_layers = []
        self.passes = 0
    def add(self, layer):
        self.layers.append(layer)
        if hasattr(layer, 'weights'):
            self.tunable_layers.append(layer)
    def set (self,*,loss, optimizer):
        self.loss = loss
        self.optimaizer = optimizer
    def regularization(self):
        self.regularized = []
        for tunable_layer in self.tunable_layers:
            self.regularized.append(self.loss.regularization_loss(tunable_layer))
        return np.sum(self.regularized)    
    def forward(self, input, label):
        self.inputs = input
        for layer in self.layers:
            layer.forward(self.inputs)
            self.inputs = layer.output
        self.last_layer_output = self.layers[len(self.layers) - 1].output
        self.calculated_loss = self.loss.calculate(self.last_layer_output, label) + self.loss.regularization_loss(self.tunable_layers) 
    def backward(self,label):
        self.loss.backward(self.last_layer_output, label)
        dinputs =  self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs       
    def batch_shuffle(self, data, label, batch):
        keys = np.array(range(data.shape[0]))
        np.random.shuffle(keys)
        data = data[keys]
        label = label[keys]
        data_split = np.array_split(data, batch)
        label_split = np.array_split(label, batch)
        return data_split, label_split
    def train(self, X, y, * , epoches:int, print_every :int = 1, batch : int= 1):
        data, label = self.batch_shuffle(X,y, batch)
        for i in range(1,  epoches + 1):
            self.forward(data[self.passes], label[self.passes])
            self.backward(label[self.passes])
            self.optimaizer.pre_update_params()
            for tunable_layer in self.tunable_layers:
                self.optimaizer.update_params(tunable_layer)
            self.optimaizer.post_update_params()
            if epoches % print_every == 0:
                    print(f'Epoches = {i},  Loss = {self.calculated_loss:.3f}, Learning_rate = {self.optimaizer.current_learning_rate :.3f}, Acc = {self.loss.accuracy(self.last_layer_output, label[self.passes]):.3f}')
            self.passes += 1
            if self.passes >= len(data):
                self.passes = 0
        self.passes = 0
    def evaluate(self, X, y):
        self.forward(X,y)
        print(f'Loss = {self.calculated_loss:.3f}, Accuracy = {self.loss.accuracy(self.last_layer_output, y):.3f}')
    def predict(self, input):
        index = input
        for layer in self.layers:
            layer.forward(index)
            index = layer.output
        l_layer_output = self.layers[len(self.layers) - 1].output
        if self.loss.__class__.__name__ == "Loss_CategoricalCrossentropy":
            self.prediction = np.argmax(l_layer_output, axis = 1)
            return self.prediction #=nb
        elif self.loss.__class__.__name__ == "Loss_BinaryCrossentropy":
            self.prediction = np.round(l_layer_output)
        else:
            return l_layer_output
    def save(self, path:str, obj):
        with open(f'{path}.pkl', 'wb') as file:
            pkl.dump(obj , file)
    @staticmethod
    def load(path):
        with open(f'{path}.pkl', "rb") as file:
            model = pkl.load(file)
        return model

















