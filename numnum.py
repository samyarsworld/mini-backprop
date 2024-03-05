class Numnum:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = _children
        self._op = _op

    def __repr__(self):
        return f"Numnum(data={self.data})"

    def __add__(self, other):
        return Numnum(self.data + other.data, (self, other), '+')
    
    def __mul__(self, other):
        return Numnum(self.data * other.data, (self, other), '*')
    
