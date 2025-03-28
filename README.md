# scrapy
 ![alt text](img.jpg)
Scrapy is a small ML library and  autograd engine inspired by karpathy's micrograd it can with vactors and tensors

# Example (autograd)
Below is a slightly contrived example showing a number of possible   supported operations:

```python
    X = graph(np.random.randn(10, 28*28))
    w = graph(np.random.randn(28*28, 10))
    b = graph(np.random.randn(1, 10))

    mul = X * w
    add = mul + b
    rel = add.ReLU()
    soft = rel.softmax()

    soft.backward()
    print(w.grad, b.grad, X.grad)
```
# Building the network
```python
    l1 = nn.Layer_Dense(28*28,200) 
    a1 = nn.Activation_ReLU()
    l2 = nn.Layer_Dense(200,100)
    a2 = nn.Activation_ReLU()
    l3 = nn.Layer_Dense(100,10)
    a3 = nn.Activation_softmax()
    loss = nn.Loss_Catagorical()
    optimizer = nn.Optimizer_ADAM(learning_rate=0.001, decay=1e-3)
    l1.forward(X.reshape(-1, 28*28)) # X is the training data 
    a1.forward(l1.output)
    l2.forward(a1.output)
    a2.forward(l2.output)
    l3.forward(a2.output)
    a3.forward(l3.output)
    loss.forward(a3.output)
    loss.backward()
    optimizer.step(l2)
    optimizer.step(l1)
```
for more information chickout "demo.ipynb"