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
            self.grad = out.value
            other.grad = out.value
        self._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(np.dot(self.value , other.value), [self, other], "*")
        def _backward():
            self.grad = other.value
            other.grad = self.value
        self._backward = _backward
        return out
    def ReLU(self):
        out = graph(np.maximum(0, self.value), [self], "ReLU")
        def _backward():
            self.grad = (self.value > 0).astype(float)
        self._backward = _backward
        return out
    def backward(self):
        pass
    def __repr__(self):
        return f'Data = {self.value}'

a = graph(-2)
b = graph(5)
c = (a + b) * graph(-4)
c.backward()
print(a.grad, b.grad)