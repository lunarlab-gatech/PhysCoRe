"""
Elasticity models: corotated, StVK, fluid and volume-preserving.
"""

from typing import *

import torch
import torch.nn as nn
from torch import Tensor

from ..abstract import Elasticity

class SigmaElasticity(Elasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)
            
        # warp svd
        U, sigma, Vh = self.svd(F)
        thredhold = 0.001
        sigma = torch.clamp_min(sigma, thredhold)
        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace
        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))
        return stress

class CorotatedElasticity(Elasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.rotation_backward_mode = 'exact'
        self.volume_j_backward_mode = 'svd'
        self.j_clamp_min = -1.0e4
        self.j_clamp_max = 1.0e4

    def configure_backward_stabilization(
        self,
        *,
        rotation_backward_mode: str = 'exact',
        volume_j_backward_mode: str = 'svd',
        j_clamp_min: float = -1.0e4,
        j_clamp_max: float = 1.0e4,
    ) -> None:
        rotation_mode = str(rotation_backward_mode).strip().lower()
        if rotation_mode in {'none', 'default'}:
            rotation_mode = 'exact'
        if rotation_mode in {'detach_rotation', 'detached'}:
            rotation_mode = 'detach'
        if rotation_mode not in {'exact', 'detach'}:
            raise ValueError(
                f'Unsupported corotated rotation backward mode {rotation_backward_mode!r} '
                '(expected "exact" or "detach").'
            )

        j_mode = str(volume_j_backward_mode).strip().lower()
        if j_mode in {'none', 'default'}:
            j_mode = 'svd'
        if j_mode in {'detached', 'frozen'}:
            j_mode = 'detach'
        if j_mode not in {'svd', 'det', 'detach'}:
            raise ValueError(
                f'Unsupported corotated J backward mode {volume_j_backward_mode!r} '
                '(expected "svd", "det", or "detach").'
            )

        self.rotation_backward_mode = rotation_mode
        self.volume_j_backward_mode = j_mode
        self.j_clamp_min = float(j_clamp_min)
        self.j_clamp_max = float(j_clamp_max)

    def _rotation_with_surrogate_backward(self, U: Tensor, Vh: Tensor) -> Tensor:
        rotation = torch.matmul(U, Vh)
        if self.rotation_backward_mode == 'detach':
            return rotation.detach()
        return rotation

    def _det3x3(self, F: Tensor) -> Tensor:
        return (
            F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1])
            - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
            + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0])
        ).view(-1, 1, 1)

    def _j_with_surrogate_backward(self, F: Tensor, sigma: Tensor) -> Tensor:
        j_svd = torch.prod(sigma, dim=1).view(-1, 1, 1)
        if self.volume_j_backward_mode == 'det':
            # Straight-through surrogate: keep the SVD-based forward J, but use
            # an explicit polynomial determinant backward instead of the SVD
            # adjoint.  torch.det backward can use inverse/LU-style formulas
            # that are ill-conditioned for singular clamped F.
            j_det = self._det3x3(F)
            return j_svd.detach() + (j_det - j_det.detach())
        if self.volume_j_backward_mode == 'detach':
            return j_svd.detach()
        return j_svd

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        F_stress = torch.nan_to_num(F, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)

        # warp svd
        U, sigma, Vh = self.svd(F_stress)

        rotation = self._rotation_with_surrogate_backward(U, Vh)
        rotation = torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
        corotated_stress = 2 * mu * torch.matmul(F_stress - rotation, F_stress.transpose(1, 2))

        J = self._j_with_surrogate_backward(F_stress, sigma)
        J = torch.nan_to_num(J, nan=1.0).clamp(self.j_clamp_min, self.j_clamp_max)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        stress = torch.nan_to_num(stress, nan=0.0).clamp(-1e6, 1e6)
        return stress
    
class FluidElasticity(Elasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        mu = 0
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)
        
        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        J = torch.nan_to_num(J, nan=1.0).clamp(-1e4, 1e4)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        stress = torch.nan_to_num(stress, nan=0.0).clamp(-1e6, 1e6)
        return stress
    
class StVKElasticity(Elasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
        
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        E = 0.5 * (FtF - I)

        stvk_stress = 2 * mu * torch.matmul(F, E)

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        volume_stress = la * J * (J - 1) * I

        stress = stvk_stress + volume_stress

        return stress
    
    
class VolumeElasticity(Elasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


        self.mode = 'taichi'

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)

        if self.mode.casefold() == 'ziran':

            #  https://en.wikipedia.org/wiki/Bulk_modulus
            kappa = 2 / 3 * mu + la

            # https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/ConstitutiveModel/EquationOfState.h
            # using gamma = 7 would have gradient issue, fix later
            gamma = 2

            stress = kappa * (J - 1 / torch.pow(J, gamma-1)) * I

        elif self.mode.casefold() == 'taichi':

            stress = la * J * (J - 1) * I

        else:
            raise ValueError('invalid mode for volume plasticity: {}'.format(self.mode))

        return stress
