import math
import numpy as np
from drloco.config.hypers import lr_scale

class Schedule(object):
    def value(self, fraction_timesteps_left):
        """
        Value of the schedule for a given timestep

        :param fraction_timesteps_left:
            (float) PPO2 does not pass a step count in to the schedule functions
             but instead a number between 0 to 1.0 indicating how much timesteps are left
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError

class LinearDecay(Schedule):
    def __init__(self, start_value, final_value):
        self.start = start_value
        self.end = final_value
        self.slope = lr_scale * (final_value - start_value)

    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.start + fraction_passed * self.slope
        # value should not be smaller then the minimum specified
        val = np.max([val, self.end])
        return val

    def __str__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'

    def __repr__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'


class LinearSchedule(LinearDecay):
    """This class is just required to be able to load models trained with the LinearSchedule
       which we later renamed to LinearDecay."""
    pass


class ExponentialSchedule(Schedule):
    def __init__(self, start_value, final_value, slope=5):
        """@param slope: determines how fast the scheduled value decreases.
           The higher the slope, the stronger is the exponential decay."""
        self.start = start_value
        self.end = final_value
        self.slope = slope
        self.difference = start_value - final_value

    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.end + np.exp(-self.slope * fraction_passed) * self.difference
        return val
        
class CosSchedule(Schedule):
    def __init__(self, start_value, final_value, num_of_period):
        self.start = start_value
        self.end = final_value
        self.total_time = 1
        self.num_of_period = num_of_period
        self.wavelength = self.total_time / self.num_of_period
        self.slope = lr_scale * (final_value - start_value)

    def cos_annealing_wave(self, x):
        theta = 2*math.pi*(x/self.wavelength)
        val = (0.5 + 0.5*math.cos(theta % math.pi))
        return val    
    
    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.slope*self.cos_annealing_wave(fraction_passed)+ self.end
        # value should not be smaller then the minimum specified
        val = np.max([val, self.end])
        return val

class CosDecaySchedule(Schedule):
    def __init__(self, start_value, final_value, num_of_period, wavelength):
        self.start = start_value
        self.end = final_value
        self.total_time = 1
        self.num_of_period = num_of_period
        self.wavelength = wavelength
        self.slope = lr_scale * (start_value - final_value)

    def cos_annealing_wave(self, x):
        period = x/self.wavelength
        if period <= self.num_of_period:
            theta = math.pi*period
            val = 0.5*(1 + math.cos(theta % math.pi))
        else:
            val = 0
        return val
    
    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        decay = fraction_timesteps_left
        val = decay*self.slope*self.cos_annealing_wave(fraction_passed) + self.end
        # value should not be smaller then the minimum specified
        val = np.max([val, self.end])
        return val
