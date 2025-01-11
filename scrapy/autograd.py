import numpy as np
#=nb

class graph:
    def __init__(self, value, nodes = [], exp = ""):
        self.value = np.array(value)
        self.nodes = nodes
        self.exp = exp
        self.grad = 0
        self._backward = lambda: None
    def __add__(self, other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(self.value + other.value, [self, other], "+")
        def _backward():
            self.grad = out.grad
            other.grad = np.sum(out.grad, axis = 0, keepdims = True) # will convert (10, 2) shaped array to (1, 2) to align the shape of the grad with the orignal value one use case is when calculating the gradient of the bias parameters 
        self._backward = _backward
        return out
    def __radd__(self, other):
        out = graph(other + self.value, [self], "C")
        def _backward():
            self.grad = other * out.grad
        self._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(np.dot(self.value , other.value), [self, other], "*")
        def _backward():
            s = np.ones(out.value.shape) * out.grad
            self.grad = np.dot(s, other.value.T)
            other.grad = np.dot(self.value.T, s) 
        self._backward = _backward
        return out
    def __sub__(self,other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(self.value - other.value, [self, other], "-")        
        def _backward():
            self.grad = out.grad
            out.grad = out.grad
        self._backward = _backward
        return out
    
    
    def __rtruediv__(self, other):
        out = graph(other /self.value, [self], "inv") # dividing a constant value by a graph
        def _backward():
            self.grad = -other/(self.value**2) * out.grad # the derivative of 1/x is -1/x^2
        self._backward = _backward
        return out
    
    def __truediv__(self, other): # dividing a graph by a constant value
        out = graph(self.value *(other**-1), [self], "div")
        def _backward():
            self.grad = (other ** -1) * out.grad
        self._backward = _backward
        return out
    def __pow__(self, other):
        out = graph(self.value**other, [self], "EXP")
        def _backward():
            self.grad = (other *(self.value**(other -1))) * out.grad
        
        self._backward = _backward
        return out
    def __neg__(self):
        out = graph(self.value * -1, [self], "-")
        def _backward():
            self.grad = -1 * out.grad
        self._backward = _backward
        return out
    def ReLU(self):
        out = graph(np.maximum(0, self.value), [self], "ReLU")
        def _backward():
            out.value[out.value > 0] = 1
            self.grad += out.grad * out.value 
        self._backward = _backward
        return out
    def e(self):#"e" stands for Euler's number eg e^self.value
        out = graph(np.exp(self.value), [self], "e")
        def _backward():
            self.grad = out.value * out.grad 
        self._backward = _backward
        return out
    def log(self): # calculates the natural logarithm of self.value
        out = graph(np.log(self.value), [self], "log")
        def _backwrad():
            self.grad = (1/self.value) * out.grad
        self._backward = _backwrad
        return out
    def backward(self):
        visited = []
        topo = []
        def build(n):
            if n not in visited:
                visited.append(n)
                for i in n.nodes:
                    topo.append(i)
                    build(i)
        build(self)
        self.grad = 1
        for i in topo:
            i._backward()
    def __repr__(self):
        return f'Data = {self.value}, Grad = {self.grad} ,exp = {self.exp}'


