import random
from numnum import Numnum

class MLP:
    def __init__(self, n_inputs, layers):
        layers = [n_inputs] + layers
        self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output
        
class Neuron:
    def __init__(self, n_input):
        self.weights = [Numnum(random.uniform(-1, 1)) for _ in range(n_input)]
        self.bias = Numnum(random.uniform(-1, 1))
    
    def __call__(self, x):
        logits = sum(a * b for (a, b) in list(zip(self.weights, x))) + self.bias
        output = logits.tanh()
        return output



n_inputs = 10
layers = [4, 3, 2, 1]
x = [2, 3, 4, 5, 6, 7, 8, 1, 2, 3]

model = MLP(n_inputs, layers)

print(model(x))