from engine import Value
import random
class Module:
    def _zeroG_(self):
        for p in self.parameters():
            p.gradient=0
    def parameters(self):
        return []
class Neuron(Module):
    def __init__(self,fan_in,nonlin=True):
        self.w=[Value(random.uniform(-1,1)) for _ in range(fan_in)]
        self.b=Value(0)
        self.nonlin=nonlin
    def __call__(self,x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return act.tanh() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]
    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'linear'} Neuron({len(self.w)})"
class Layer(Module):
    def __init__(self,fan_in,fan_out,**kwargs):
        self.
        
        