"""
Constitutive models — thin wrapper around omniphysgs original models.

Imports the physical constitutive models from src/constitutive_models/ which use
warp-based SVD for correct deformation gradient decomposition.
"""

import torch.nn as nn

# Re-export original omniphysgs constitutive models
from physcore.constitutive.physical_constitutive_models.elasticity import (
    CorotatedElasticity,
    StVKElasticity,
    SigmaElasticity,
    FluidElasticity,
)
from physcore.constitutive.physical_constitutive_models.plasticity import (
    IdentityPlasticity,
    DruckerPragerPlasticity,
    VonMisesPlasticity,
    SigmaPlasticity,
)


# ─────────────── Model Registry ───────────────────────────────────────────────

ELASTICITY_REGISTRY = {
    'CorotatedElasticity': CorotatedElasticity,
    'StVKElasticity': StVKElasticity,
    'SigmaElasticity': SigmaElasticity,
    'FluidElasticity': FluidElasticity,
}

PLASTICITY_REGISTRY = {
    'IdentityPlasticity': IdentityPlasticity,
    'DruckerPragerPlasticity': DruckerPragerPlasticity,
    'VonMisesPlasticity': VonMisesPlasticity,
    'SigmaPlasticity': SigmaPlasticity,
}


def get_elasticity_model(name: str, **kwargs) -> nn.Module:
    if name not in ELASTICITY_REGISTRY:
        raise ValueError(f"Unknown elasticity: {name}. Available: {list(ELASTICITY_REGISTRY.keys())}")
    return ELASTICITY_REGISTRY[name](**kwargs)


def get_plasticity_model(name: str, **kwargs) -> nn.Module:
    if name not in PLASTICITY_REGISTRY:
        raise ValueError(f"Unknown plasticity: {name}. Available: {list(PLASTICITY_REGISTRY.keys())}")
    return PLASTICITY_REGISTRY[name](**kwargs)
