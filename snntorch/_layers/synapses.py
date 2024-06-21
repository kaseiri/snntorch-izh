import torch
import torch.nn as nn


class Synapses(nn.Module):
    def __init__(self, num_inputs, num_outputs, tau):
        super().__init__()
        self.tau = tau
        self.syn = None
        self.weights = nn.Parameter(0.1 * (2 * torch.rand(num_inputs, num_outputs) - 1))

    def reset_syn(self):
        self.syn = None

    def forward(self, in_spikes, mem, rev):
        if self.syn is None or self.syn.size(0) != in_spikes.size(0):
            self.syn = torch.zeros(in_spikes.size(0), in_spikes.size(1),
                                   self.weights.size(1), device=in_spikes.device)

        decay_factor = 1 - 1 / self.tau
        spike_contribution = in_spikes.unsqueeze(-1).expand(-1, -1, self.weights.size(1))
        self.syn = self.syn * decay_factor + spike_contribution

        conductivity = self.syn * self.weights.unsqueeze(0)
        voltage = rev.unsqueeze(2) - mem.unsqueeze(1)
        current = conductivity * voltage

        return current.sum(dim=1)
