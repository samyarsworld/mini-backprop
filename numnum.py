import math
import torch
class Numnum:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._children = _children
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Numnum(data={self.data})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            val = other
        elif isinstance(other, Numnum):
            val = other.data
        else:
            assert("Values most be a number or a Numnum object.")

        output = Numnum(self.data + val, (self, other), '+')
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward
        return output
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            val = other
        elif isinstance(other, Numnum):
            val = other.data
        else:
            assert("Values most be a number or a Numnum object.")
        output = Numnum(self.data * val, (self, other), '*')
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output

    def __pow__(self, n):
        if isinstance(n, (int)):
            assert("Power value most be an integer or float.")

        # Fast power calculator for integers
        def calcPow(val, n):
            if n == 0:
                return 1
            if n % 2 == 0:
                return calcPow(val * val, n // 2)
            return val * calcPow(val * val, (n - 1) // 2)

        res = calcPow(self.data, n) if isinstance(n, int) else self.data ** n
        output = Numnum(res, (self, ), '**')

        def _backward():
            self.grad += (n * self.data ** (n - 1)) * output.grad
        
        output._backward = _backward
        return output
    
    def relu(self):
        output = Numnum(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output
    
    def tanh(self):
        tanh_val = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        output = Numnum(tanh_val, (self,), 'tanh')

        def _backward():
            self.grad += (1 - tanh_val ** 2) * output.grad
        output._backward = _backward

        return output
    
    def backward(self):
        self.grad = 1.0
        
        visited = set()
        graph = []
        def sortTopological(node):
            if node not in visited:
                for child in node._children:
                    sortTopological(child)
                graph.append(node)
        sortTopological(self)

        for node in graph[::-1]:
            node._backward()
            
            
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1


