"""
Plasticity models: Drucker-Prager, von Mises, and an identity no-op.
"""

import math
from typing import *
import torch
import torch.nn as nn
from torch import Tensor

from ..abstract import Plasticity

class DruckerPragerPlasticity(Plasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.register_buffer('friction_angle', torch.Tensor([25.0]))
        self.register_buffer('cohesion', torch.Tensor([0.0]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:

        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan
        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F
    
class IdentityPlasticity(Plasticity):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        return F
    
    
class SigmaPlasticity(Plasticity):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        J = torch.clamp(J, min=0.05, max=1.2)

        Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
        F = torch.diag_embed(Je_1_3)
        return F
    
    
class VonMisesPlasticity(Plasticity):
    def __init__(self, sigma_y: float = 1.0e3, log_E: float = math.log(2.0e6), nu: float = 0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.tensor([float(log_E)]))
        self.register_buffer('nu', torch.tensor([float(nu)]))
        self.register_buffer('sigma_y', torch.tensor([float(sigma_y)]))

        # Optional yield-event tracking (off by default; opt-in via enable_yield_tracking).
        # Used by the augmenter to record which particles actually plastically yielded.
        self._track_yield: bool = False
        self.ever_yielded: Optional[Tensor] = None
        self.total_yield_events: int = 0
        self.frames_with_yield: int = 0

    def enable_yield_tracking(self, num_particles: int) -> None:
        device = self.sigma_y.device
        self._track_yield = True
        self.ever_yielded = torch.zeros(int(num_particles), dtype=torch.bool, device=device)
        self.total_yield_events = 0
        self.frames_with_yield = 0

    def disable_yield_tracking(self) -> None:
        self._track_yield = False

    def reset_yield_tracking(self) -> None:
        if self.ever_yielded is not None:
            self.ever_yielded.zero_()
        self.total_yield_events = 0
        self.frames_with_yield = 0

    def get_yield_stats(self) -> Dict[str, object]:
        if self.ever_yielded is None:
            return {
                "any_particle_yielded": False,
                "num_particles_ever_yielded": 0,
                "fraction_particles_ever_yielded": 0.0,
                "total_yield_events": 0,
                "frames_with_yield": 0,
            }
        n_yielded = int(self.ever_yielded.sum().item())
        n_total = int(self.ever_yielded.numel())
        return {
            "any_particle_yielded": bool(n_yielded > 0),
            "num_particles_ever_yielded": n_yielded,
            "fraction_particles_ever_yielded": float(n_yielded) / max(n_total, 1),
            "total_yield_events": int(self.total_yield_events),
            "frames_with_yield": int(self.frames_with_yield),
        }

    def get_per_particle_yield_mask(self) -> Optional[Tensor]:
        if self.ever_yielded is None:
            return None
        return self.ever_yielded.detach().clone()

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:


        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        sigma_y = self.sigma_y

        mu = E / (2 * (1 + nu))
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan

        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)

        F = torch.where(cond_yield, yield_F, F)

        if self._track_yield and self.ever_yielded is not None:
            with torch.no_grad():
                flat = cond_yield.view(-1)
                if flat.shape[0] == self.ever_yielded.shape[0]:
                    self.ever_yielded |= flat
                    n_events = int(flat.sum().item())
                    self.total_yield_events += n_events
                    if n_events > 0:
                        self.frames_with_yield += 1

        return F
