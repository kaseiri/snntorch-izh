from .neurons import SpikingNeuron
import torch
from torch import nn

class SCIZ(SpikingNeuron):
    # class attribute for neuron types
    neuron_types = {
            'RS': {
                'a1': 1.0, 'a2': -0.210, 'a3': 0.019,
                'b1': -1.0 / 32.0, 'b2': 1.0 / 32.0, 'b3': 0.0,
                'c': 0.105, 'd': 0.412,
                'v_thr': 0.7
            },
            'IB': {
                'a1': 1.0, 'a2': -1/4.0, 'a3': 0.043,
                'b1': 1.0 / 128.0, 'b2': 1.0 / 64.0, 'b3': 0.0,
                'c': 0.152, 'd': 0.164,
                'v_thr': 1.000
            },
            'CH': {
                'a1': 4.0, 'a2': -0.660, 'a3': 0.077,
                'b1': 1.0 / 128.0, 'b2': 1.0 / 32.0, 'b3': 0.0,
                'c': 0.158, 'd': 0.295,
                'v_thr': 0.645
            }
        }

    def __init__(
        self,
        neuron_type='RS',
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        output=False,
        state_quant=None
    ):
        # SC-IZ normalized parameters
        self.neuron_type = neuron_type


        # set parameters according to neuron type
        self.__dict__.update(SCIZ.neuron_types[self.neuron_type])

        super().__init__(
            threshold=self.v_thr,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            output=output,
            state_quant=state_quant
        )

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

        if self.state_quant:
            self.mem, self.rec = self.state_quant(self.mem), self.state_quant(self.rec)

        incr_mem = self.a1 * self.mem * self.mem + self.a2 * self.mem - self.a3 * self.rec + input_
        incr_rec = self.b1 * self.mem - self.b2 * self.rec + self.b3

        self.reset = self.mem_reset(self.mem)  # detached reset signal
        reset_mem = self.reset * (-self.mem + self.c - incr_mem)
        reset_rec = self.reset * (self.d - incr_rec)

        self.mem = self.mem + incr_mem + reset_mem
        self.rec = self.rec + incr_rec + reset_rec

        spk = self.fire(self.mem)

        #self.mem = torch.where(spk.bool(), self.threshold + 1e-4, self.mem)

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
