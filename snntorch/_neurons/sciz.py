from .neurons import SpikingNeuron
import torch
from torch import nn

class SCIZ(SpikingNeuron):
    def __init__(
        self,
        threshold=0.7,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        output=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            output,
        )

        # SC-IZ normalized parameters
        self.a1, self.a2, self.a3 = 1.0, -0.210, 0.019
        self.b1, self.b2, self.b3 = -1.0 / 32.0, 1.0 / 32.0, 0.0
        self.c, self.d = 0.105, 0.412

        self._init_mem()

    def _init_mem(self):
        mem, rec = torch.zeros(0), torch.zeros(0)
        self.register_buffer("mem", mem, False)
        self.register_buffer("rec", rec, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.rec = torch.zeros_like(self.rec, device=self.rec.device)
        return self.mem, self.rec

    def forward(self, input_, mem=None, rec=None):
        if mem is not None or rec is not None:
            self.mem, self.rec = mem, rec

        if self.init_hidden and (mem is not None or rec is not None):
            raise TypeError(
                "`mem` or `rec` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape or not self.rec.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
            self.rec = torch.zeros_like(input_, device=self.rec.device)

        # reset previously spiking neurons
        self.reset = self.mem_reset(self.mem)  # detached reset signal
        self.mem = self.reset * (-self.mem + self.c - incr_mem)
        self.rev = self.reset * (self.d - incr_rec)

        # update state
        scale = 1
        incr_mem = self.a1 * self.mem * self.mem + self.a2 * self.mem - self.a3 * self.rec + input_
        incr_rec = self.b1 * self.mem - self.b2 * self.rec + self.b3
        self.mem = self.mem + scale * incr_mem
        self.rev = self.rev + scale * incr_rev

        # spike
        spk = self.fire(self.mem)

        # reset currently spiking neurons
        post_reset = spk - self.reset  # detached reset signal
        self.mem = post_reset * (-self.mem + self.c - incr_mem)
        self.rev = post_reset * (self.d - incr_rec)

        if self.output:
            return spk, self.mem, self.rec
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem, self.rec

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SCIZ):
                cls.instances[layer].mem.detach_()
                cls.instances[layer].rec.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SCIZ):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
                cls.instances[layer].rec = torch.zeros_like(
                    cls.instances[layer].rec,
                    device=cls.instances[layer].rec.device,
                )
