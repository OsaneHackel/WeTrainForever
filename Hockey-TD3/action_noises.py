# I asked ChatGPT to verify my implementation of Colored noise and incorporated its corrections

import torch
import torch.nn as nn
import numpy as np


# Ornstein-Uhlenbeck noise
class OUNoise:
    # temporally correlated noise
    # previous noise carries over
    # theta for mean reversion
    # randomness added on top
    def __init__(self, shape, device, 
                 seed=None,
                 theta=0.15, dt=1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self._device = device
        self.previous_noise = torch.zeros(self._shape)
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(seed)
    
    def __call__(self):
        noise = (self.previous_noise 
                 + self._theta * (-self.previous_noise) * self._dt
                 + np.sqrt(self._dt) * torch.randn(self._shape, 
                                                   device=self._device,
                                                   generator=self._generator))
        self.previous_noise = noise
        return noise
    
    def reset(self):
        self.previous_noise = torch.zeros(self._shape, device=self._device)

# Pink (beta=1)/Colored noise
class ColoredNoise:
    def __init__(self, 
                 beta: float, # 0=white noise (temp. uncorrelated), 2=red noise (temp. correlated)
                 act_dim: int,
                 episode_length: int, 
                 device,
                 rng: np.random.Generator = None):
        self._beta = beta   # color param (1.0 = pink noise)
        self._act_dim = act_dim
        self._episode_length = episode_length  # 250 = NORMAL
        self._device = device
        self._rng = rng or np.random.default_rng()
        self._counter = 0
        self._noise_signal = None      # (episode_length, act_dim)
        self.reset() 
        
    @staticmethod
    def _generate_colored_gaussian_noise(beta, size, rng):
        """
        Generates size many noise values that are Gaussian-distributed
        overall but for beta>0 they are temporally correlated (noise
        drifts smoothly instead of "moving up and down" independently)

        Returns: numpy array of shape (size, ) with var=1, mean=0
        """
        # frequencies 
        n_freqs = size // 2 + 1
        freqs = np.fft.rfftfreq(size, d=1.0)
        
        # sample Fourier coefficients [$a_k + i b_k$] acc. to std Gaussian 
        coeff = rng.standard_normal(n_freqs) + 1j * rng.standard_normal(n_freqs)
        
        # Enforce "real signal" constraints: DC and Nyquist must be purely real.
        coeff[0] = coeff[0].real + 0j               # DC imag = 0 
        if size % 2 == 0: 
            coeff[-1] = coeff[-1].real + 0j         # Nyquist imag = 0
            
        # scaling their impact on the signal by f^{-\beta / 2}
        scale = np.ones(n_freqs) 
        scale[0] = 0.0                               # remove DC -> zero mean
        # avoid divide-by-zero at f=0 by starting from index 1
        scale[1:] = freqs[1:] ** (-beta / 2.0)
        shaped_coeff = coeff * scale 
        
        # transform back -> time-domain signal
        x = np.fft.irfft(shaped_coeff, n=size)
        
        # normalize s.t. different betas have comparable magnitude
        x = x - x.mean()
        std = x.std()
        if std > 0:
            x /= std
            
        return x.astype(np.float32)
    
    def reset(self):
        """
        Should be called at episode start.
        Generates a fresh episode of colored noise that
        is independent for each action dimension (but correlated within).
        """
        self._counter = 0
        # generate independent colored noise for each act dim
        noises = np.stack([
            self._generate_colored_gaussian_noise(self._beta,
                                                  self._episode_length,
                                                  self._rng)
            for d in range(self._act_dim)
        ], axis=1) # (episode_length, act_dim)
        self._noise_signal = torch.from_numpy(noises).to(self._device)
        
    def __call__(self)-> torch.Tensor:
        """
        Gets noise for the current step
        Returns: noise of shape (act_dim, )
        """
        idx = min(self._counter, self._episode_length -1)
        noise = self._noise_signal[idx]
        self._counter += 1
        return noise
