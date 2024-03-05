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
        output = Numnum(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = output.grad
            other.grad = output.grad
        output._backward = _backward
        return output
    
    def __mul__(self, other):
        output = Numnum(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * output.grad
            other.grad = self.data * output.grad
        output._backward = _backward
        return output

    def __pow__(self, n):
        def calcPow(val, n):
            if n == 0:
                return 1
            if n % 2 == 0:
                return calcPow(val * val, n // 2)
            return val * calcPow(val * val, (n - 1) // 2)

        output = Numnum(calcPow(self.data, n), (self, ), '**')

        # def _backward():
        #     self.grad = other.data * output.grad

        # output._backward = _backward
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
            
            



a = Numnum(10)
b = Numnum(15)
d = Numnum(2)
f = Numnum(3)

c = a + b
e = c * d
g = f * e

g.grad = 1.0
g.backward()

print(f.grad)

