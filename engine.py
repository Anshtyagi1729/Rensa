# remising the simpler times again
import math
from operator import sub


class Value:
    def __init__(self, data: float, _children=(), _op=""):
        self.data = data
        self.gradient = 0.0
        self.children = set(_children)
        self.ops = _op

    def _backward(self):
        pass

    def __repr__(self) -> str:
        return f"data is :{self.data} \ngrad is {self.gradient} "

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "NO"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.gradient += (other * self.data ** (other - 1)) * out.gradient

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.gradient += (out.data > 0) * out.gradient

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.gradient += (1 - out.data**2) * out.gradient

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        vis = set()

        def build_topo(node):
            if node not in vis:
                vis.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()

    # apparantly these are called dunder functions

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other - self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1


def main():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    out = a * b + c
    out.backward()
    print(a)  # gradient should be 3.0
    print(b)  # gradient should be 2.0
    print(c)  # gradient should be 1.0


if __name__ == "__main__":
    main()
