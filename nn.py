import numpy as np
import pickle as pkl


#=nb

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
    def prediction(self):
        return np.argmax(self.output, -1) 
        
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    def prediction(self):
        return (self.output > 0.5) * 1
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    def prediction(self):
        return self.output
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
        
        #dvalues[dvalues == 0] = 1e-4
        self.dinputs = -y_true / dvalues 
        
        self.dinputs = self.dinputs / samples
        return self.dinputs
        
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
        return self.dinputs
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
class Loss_MeanAbsoluteError(Loss):  
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
class Model:
    def __init__(self):
        self.layers = []
        self.tunableLayer = []
    def add(self, layers):
        if hasattr(layers, "weights"):
            self.layers.append(layers)
            self.tunableLayer.append(layers)
        else :
            self.layers.append(layers)
    
    def set(self, *, Loss, Optimizer):
        self.Loss = Loss
        self.Optimizer = Optimizer

    def forward(self, train_data):
        data = train_data
        for i in self.layers:
            i.forward(data)
            data = i.output 
        return self.layers[len(self.layers) -1]
    def _backward(self, dinputs, target):
        derivative = self.Loss.backward(dinputs, target)
        for i in reversed(self.layers):
            i.backward(derivative)
            derivative = i.dinputs
    def step(self):
        self.Optimizer.pre_update_params()
        for i in self.tunableLayer:
            self.Optimizer.update_params(i)
            self.Optimizer.post_update_params()
    def _batch(self, X, y , shuffle: bool = True, batch = 1):
        keys = np.array(range(X.shape[0]))
        if shuffle:
            X = X[keys]
            y = y[keys]
        data_split = np.array_split(X, batch)
        label_split = np.array_split(y, batch)
        return data_split, label_split
    def train(self,X, y, *,batch = 1, shuffle = True, print_every = 1):
            data, label = self._batch(X,y, shuffle= shuffle, batch = batch)
            for batch in range(batch):
                pred = self.forward(data[batch])
                loss = self.Loss.calculate(pred.output, label[batch])
                if batch % print_every == 0:
                    current = len(data[0]) * (batch + 1)
                    print(f'Loss : {loss :.4f}     [{current}/ {len(X)}]')
                self._backward(pred.output, label[batch])
                self.step()
    def test(self, X,y,*, batch:int = 1, shuffle = True):
        data, label = self._batch(X, y, batch= batch, shuffle = shuffle)
        test_loss , correct,total_pred = 0,0,0
        for batch in range(batch):
            pred = self.forward(data[batch])
            test_loss += self.Loss.calculate(pred.output, label[batch])
            correct += np.sum(pred.prediction() == label[batch])
        test_loss /= batch
        correct /= len(X)
        
        print(f"Test LOSS : {test_loss :.4f}    Accuracy = {correct * 100 :.1f} ")
    
    def predict(self, img):
        pred = self.forward(img)
        return pred.prediction()

    def save(self, path:str, obj):
        with open(f'{path}.pkl', 'wb') as file:
            pkl.dump(obj , file)
    @staticmethod
    def load(path):
        with open(f'{path}.pkl', "rb") as file:
            model = pkl.load(file)
        return model




