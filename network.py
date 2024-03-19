import random
from lib import Numnum

class MLP:
    def __init__(self, n_inputs, layers):
        layers = [n_inputs] + layers
        self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    

class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
        
class Neuron:
    def __init__(self, n_input):
        self.weights = [Numnum(random.uniform(-1, 1)) for _ in range(n_input)]
        self.bias = Numnum(random.uniform(-1, 1))
    
    def __call__(self, x):
        logits = sum(a * b for (a, b) in list(zip(self.weights, x))) + self.bias
        output = logits.tanh()
        return output
    
    def parameters(self):
        return self.weights + [self.bias]
