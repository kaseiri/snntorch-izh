import torch
from .neurons import SpikingNeuron


class Izhikevich(SpikingNeuron):
    def __init__(
        self,
        a=0.02,
        b=2.0,
        c=-65,
        d=8.0,  # excitatory, 2 for inhibitory
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

        # inits SpikingNeuron
        reset_mechanism = 'subtract'
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

        # model parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.mem_rest = mem_rest
        self.u_rest = u_rest
        self.dt = 0.5

        # add u and v to nn.Module.register_buffer
        self._init_mem()

    def _init_mem(self):
        mem = torch.zeros(0)
        u = torch.zeros(0)
        self.register_buffer("mem", mem, False)
        self.register_buffer("u", u, False)

    def reset_mem(self):
        self.mem = torch.full_like(self.mem, self.mem_rest, device=self.mem.device)
        self.u = torch.full_like(self.u, self.u_rest, device=self.u.device)
        return self.mem, self.u

    def forward(self, input_):
        if not self.mem.shape == input_.shape:
            self.mem = torch.full_like(input_, self.mem_rest, device=self.mem.device)
        if not self.u.shape == input_.shape:
            self.u = torch.full_like(input_, self.u_rest, device=self.u.device)

        # get SpikingNeuron reset signal
        self.reset = self.mem_reset(self.mem)

        # run izh equations and reset mechanism
        self.update_izhikevich(input_,)

        # get spike from SpikingNeuron
        spk = self.fire(self.mem)

        if self.init_hidden:
            return spk
        else:
            return spk, self.mem

    def update_izhikevich(self, input_):
        # Temporary copies for calculations
        temp_mem = self.mem.clone()
        temp_u = self.u.clone()

        # Calculate updates only for neurons that are not resetting
        not_resetting = self.reset == 0
        if not_resetting.any():
            dmem = 0.04 * temp_mem[not_resetting]**2 + 5 * temp_mem[not_resetting] + 140 - temp_u[not_resetting] + input_[not_resetting]
            du = self.a * (self.b * temp_mem[not_resetting] - temp_u[not_resetting])
            temp_mem[not_resetting] += self.dt * dmem
            temp_u[not_resetting] += self.dt * du

        # Apply reset conditions
        temp_mem = torch.where(self.reset > 0, torch.full_like(self.mem, self.c), temp_mem)
        temp_u = torch.where(self.reset > 0, self.u + self.d, temp_u)

        # Clamp membrane potential to prevent any excessive overshoot
        temp_mem = torch.clamp(temp_mem, max=self.threshold + 0.001)

        # Update the actual class variables after all calculations
        self.mem = temp_mem
        self.u = temp_u

#    def update_izhikevich(self, input_, mem, u):
#        # first half of time step
#        dmem = 0.04 * mem**2 + 5 * mem + 140 - self.u + input_
#        du = self.a * (self.b * mem - self.u)
#        mem = mem + self.dt * dmem
#        u = u + self.dt * du
#
#        # second half of time step
#        dmem = 0.04 * mem**2 + 5 * mem + 140 - self.u + input_
#        du = self.a * (self.b * mem - self.u)
#        mem = mem + self.dt * dmem
#        u = u + self.dt * du
#
#        # reset
#        mem = torch.where(self.reset > 0, torch.full_like(mem, self.c), mem)
#        u = torch.where(self.reset > 0, u + self.d, u)
#
#        # clamp membrane potential to threshold
#        mem = torch.clamp(mem, max=self.threshold+0.01)
#        return mem, u
