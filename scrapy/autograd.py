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
            self.grad += out.grad
            other.grad += out.grad
        self._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(np.dot(self.value , other.value), [self, other], "*")
        def _backward():
            self.grad += np.dot(out.grad, other.value.T)
            other.grad += np.dot(self.value.T, other.grad)
        self._backward = _backward
        return out
    def __sub__(self,other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(self.value - other.value, [self, other], "-")
        return out
    def __rtruediv__(self, other):
        out = graph(other /self.value, [self], "inv") # dividing a constant value by a graph
        def _backward():
            self.grad += -other/(self.value**2) * out.grad # the derivative of 1/x is -1/x^2
        self._backward = _backward
        return out
    def __truediv__(self, other):
        out = graph(self.value *(other**-1), [self], "div")
        def _backward():
            self.grad += (other ** -1) * out.grad
        self._backward = _backward
        return out
    def __pow__(self, other):
        out = graph(self.value**other, [self], "EXP")
        def _backward():
            self.grad += (other *(self.value**(other -1))) * out.grad
        
        self._backward = _backward
        return out
        
    def ReLU(self):
        out = graph(np.maximum(0, self.value), [self], "ReLU")
        def _backward():
            out.value[out.value > 0] = 1
            self.grad = out.grad * out.value 
        self._backward = _backward
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
        return f'Data = {self.value}, Grad = {self.grad}'


