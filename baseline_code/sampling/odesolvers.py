import abc

import torch
import numpy as np
from typing import Callable

class Registry:
    def __init__(self, managed_thing: str):
        """
        Create a new registry.

        Args:
            managed_thing: A string describing what type of thing is managed by this registry. Will be used for
                warnings and errors, so it's a good idea to keep this string globally unique and easily understood.
        """
        self.managed_thing = managed_thing
        self._registry = {}

    def register(self, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in self._registry:
                warnings.warn(f"{self.managed_thing} with name '{name}' doubly registered, old class will be replaced.")
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get_by_name(self, name: str):
        """Get a managed thing by name."""
        if name in self._registry:
            return self._registry[name]
        else:
            raise ValueError(f"{self.managed_thing} with name '{name}' unknown.")

    def get_all_names(self):
        """Get the list of things' names registered to this registry."""
        return list(self._registry.keys())


ODEsolverRegistry = Registry("ODEsolver")


class ODEsolver(abc.ABC):
    

    def __init__(self, ode, VF_fn):
        super().__init__()
        self.ode = ode        
        self.VF_fn = VF_fn
        

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@ODEsolverRegistry.register('euler')
class EulerODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
        vectorfield = self.VF_fn(x,t,y)
        x = x + vectorfield*dt
        
        return x


@ODEsolverRegistry.register('midpoint')
class MidpointODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
       
        x = x + dt*self.VF_fn(x+dt/2*self.VF_fn(x,t,y), t+dt/2, y)
        
        return x
    
@ODEsolverRegistry.register('heun')
class HeunODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
        current_vectorfield = self.VF_fn(x,t,y)
        x_next = x + dt * current_vectorfield
        x = x + dt/2 *(current_vectorfield+self.VF_fn(x_next,t+dt, y))
        
        return x
    
