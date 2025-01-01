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
            other.grad = out.grad
        self._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, graph) else graph(other)
        out = graph(np.dot(self.value , other.value), [self, other], "*")
        def _backward():
            self.grad = other.value * out.grad
            other.grad = self.value * out.grad
        self._backward = _backward
        return out
    def ReLU(self):
        out = graph(np.maximum(0, self.value), [self], "ReLU")
        def _backward():
            self.grad = (self.value > 0).astype(float)
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
        return f'Data = {self.value}'

x = graph(-2)
w = graph(5)
b = graph(-4)
c = (x + w) * b
print(c)
c.backward()
print(b.grad, w.grad, x.grad)