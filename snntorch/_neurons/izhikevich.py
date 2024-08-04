import torch
import torch.nn as nn
from .neurons import SpikingNeuron

class Izhikevich(SpikingNeuron):
    def __init__(
        self,
        num_neurons,  # Number of neurons passed during initialization
        a=0.02,
        b=0.20,
        c=-65,
        d=8.0,
        mem_rest=-70,
        u_rest=-14,
        threshold=35.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):

        reset_mechanism = 'zero'
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self.num_neurons = num_neurons  # Number of neurons known at initialization
        self.a = nn.Parameter(torch.full((num_neurons,), a, dtype=torch.float))
        self.b = nn.Parameter(torch.full((num_neurons,), b, dtype=torch.float))
        self.c = nn.Parameter(torch.full((num_neurons,), c, dtype=torch.float))
        self.d = nn.Parameter(torch.full((num_neurons,), d, dtype=torch.float))
        self.mem_rest = mem_rest
        self.u_rest = u_rest
        self.dt = 1
        self.state_function = self.izhikevich_state_function

        self._init_mem()

    def _init_params(self, num_neurons, device):
        self.num_neurons = num_neurons
        self.a = nn.Parameter(torch.full((num_neurons,), self.a_init, dtype=torch.float, device=device))
        self.b = nn.Parameter(torch.full((num_neurons,), self.b_init, dtype=torch.float, device=device))
        self.c = nn.Parameter(torch.full((num_neurons,), self.c_init, dtype=torch.float, device=device))
        self.d = nn.Parameter(torch.full((num_neurons,), self.d_init, dtype=torch.float, device=device))

    def _init_mem(self):
        self.register_buffer("mem", torch.zeros(0))
        self.register_buffer("u", torch.zeros(0))

    def reset_mem(self):
        self.mem = torch.full_like(self.mem, self.mem_rest, device=self.mem.device)
        self.u = torch.full_like(self.u, self.u_rest, device=self.u.device)
        return self.mem, self.u

    def forward(self, input_, mem=None):
        batch_size, num_neurons = input_.shape
        device = input_.device

        if not self.mem.shape == input_.shape:
            self.mem = torch.full((batch_size, num_neurons), self.mem_rest, device=device)
            self.u = torch.full((batch_size, num_neurons), self.u_rest, device=device)

        if self.init_hidden and mem is not None:
            raise TypeError("`mem` should not be passed as an argument while `init_hidden=True`")

        self.reset = self.mem_reset(self.mem)
        self.mem = self.mem - self.reset * (self.mem - self.c)
        self.u = self.u + self.reset * self.d
        self.mem, self.u = self.state_function(input_)
        spk = self.fire(self.mem)
        self.mem = torch.clamp(self.mem, max=self.threshold + 1e-4)

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

    def izhikevich_state_function(self, input_):
        dmem = (0.04 * self.mem ** 2 + 5 * self.mem + 140 - self.u + input_)
        du = self.a * (self.b * self.mem - self.u)
        new_mem = self.mem + self.dt * dmem
        new_u = self.u + self.dt * du
        return new_mem, new_u

    @classmethod
    def detach_hidden(cls):
        for instance in cls.instances:
            if isinstance(instance, Izhikevich):
                instance.mem = instance.mem.detach()
                instance.u = instance.u.detach()

    @classmethod
    def reset_hidden(cls):
        for instance in cls.instances:
            if isinstance(instance, Izhikevich):
                instance.mem = torch.full_like(instance.mem, instance.mem_rest, device=instance.mem.device)
                instance.u = torch.full_like(instance.u, instance.u_rest, device=instance.u.device)
