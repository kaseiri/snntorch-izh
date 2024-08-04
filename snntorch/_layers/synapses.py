import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class Synapses(nn.Module):
    def __init__(self, num_inputs, num_outputs, tau):
        super().__init__()
        self.tau = tau
        self.syn = None
        self.weights = nn.Parameter(torch.empty(num_inputs, num_outputs))
        nn.init.uniform_(self.weights, a=0, b=0.01)
        print("pre", self.weights)
        #self.initialize_weights_uniform_exp(a=5e-7, b=5e-2)
        P.register_parametrization(self, "weights", PositiveWeights())
        print("post", self.weights)

    def initialize_weights_uniform_exp(self, a, b):
        with torch.no_grad():
            self.weights.uniform_(a, b).log_()
        
    def reset_syn(self):
        self.syn = None

    def forward(self, in_spikes, mem):
        if self.syn is None or self.syn.size(0) != in_spikes.size(0):
            self.syn = torch.zeros(in_spikes.size(0), in_spikes.size(1),
                                   self.weights.size(1), device=in_spikes.device)

        decay_factor = 1 - 1 / self.tau
        spike_contribution = in_spikes.unsqueeze(-1).expand(-1, -1, self.weights.size(1))
        self.syn = self.syn * decay_factor + spike_contribution

        conductivity = self.syn * self.weights.unsqueeze(0)
        voltage = - mem.unsqueeze(1)
        current = conductivity * voltage

        return current.sum(dim=1)


class PositiveWeights(nn.Module):
    def forward(self, X):
        #return X.exp()
        return torch.clamp(X, min=0)
        

