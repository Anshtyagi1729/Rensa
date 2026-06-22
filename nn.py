import random

from engine import Value


class Module:
    def _zeroG_(self):
        for p in self.parameters():
            p.gradient = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, fan_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(fan_in)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'linear'} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, fan_in, fan_out, **kwargs):
        self.neurons = [Neuron(fan_in, **kwargs) for _ in range(fan_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, fan_in, fan_outs):
        # so the fan in tell the input size and fan_outs is the arr of ouputs so that is layer sizes
        # we do it better but its better if i dont
        size = [fan_in] + fan_outs
        self.layers = [
            Layer(size[i], size[i + 1], nonlin=i != (len(fan_outs) - 1))
            for i in range(fan_outs)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    # i hate writing the repr
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
