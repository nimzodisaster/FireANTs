from typing import List, Optional, Union
import torch
from torch import nn
from torch.optim import SGD, Adam
from fireants.io.image import BatchedImages
from fireants.registration.rigid import RigidRegistration
import logging

logger = logging.getLogger(__name__)

class SimilarityRegistration(RigidRegistration):
    """
    Similarity registration class (Rigid + Isotropic Scaling).
    Includes safeguards against scale explosion.
    """

    def __init__(self, scales: List[float], iterations: List[int],
                 fixed_images: BatchedImages, moving_images: BatchedImages,
                 loss_type: str = "cc",
                 optimizer: str = 'Adam', optimizer_params: dict = {},
                 optimizer_lr: float = 1e-2, # Reduced default LR slightly
                 scale_only: bool = False, 
                 max_scale_change: float = 0.5, # New: Limit log-scale to +/- 0.5 (approx 0.6x to 1.65x)
                 scale_lr_factor: float = 0.1,  # New: Scale learns 10x slower than translation
                 **kwargs
                 ) -> None:

        # 1. Initialize Parent
        kwargs['scaling'] = False
        super().__init__(scales, iterations, fixed_images, moving_images,
                         loss_type=loss_type, optimizer=optimizer,
                         optimizer_params=optimizer_params, optimizer_lr=optimizer_lr,
                         **kwargs)

        self.max_scale_change = max_scale_change

        # 2. Define Isotropic Scale Parameter
        device = fixed_images.device
        self.logscale = nn.Parameter(torch.zeros((self.opt_size, 1), device=device, dtype=self.dtype))

        # 3. Parameter Groups
        # We must separate scale from rigid params to give it a lower learning rate
        
        if scale_only:
            # Freeze Rigid
            self.rotation.requires_grad_(False)
            self.transl.requires_grad_(False)
            
            # Optimize only scale, but with the reduced factor
            param_groups = [
                {'params': self.logscale, 'lr': optimizer_lr * scale_lr_factor}
            ]
            logger.info("Similarity Registration: Optimizing SCALE ONLY (Rigid frozen)")
        else:
            # Optimize both, but separate groups
            param_groups = [
                {'params': [self.rotation, self.transl], 'lr': optimizer_lr},
                {'params': self.logscale, 'lr': optimizer_lr * scale_lr_factor}
            ]
            logger.info("Similarity Registration: Optimizing Scale (Slow), Rotation, and Translation")

        # 4. Initialize Optimizer with groups
        if optimizer == 'SGD':
            self.optimizer = SGD(param_groups, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam(param_groups, **optimizer_params)

    def get_rigid_matrix(self, homogenous=True):
        """
        Compute the transformation matrix using isotropic scaling.
        """
        rigidmat = self.get_rotation_matrix() # [N, dim+1, dim+1]
        
        # --- SAFETY LATCH: CLAMPING ---
        # Prevents the optimizer from zooming to infinity or zero.
        # Clamping in the forward pass allows gradients to flow until the limit is hit.
        clamped_logscale = torch.clamp(self.logscale, -self.max_scale_change, self.max_scale_change)
        scale = torch.exp(clamped_logscale)[..., None]
        # ------------------------------

        matclone = rigidmat.clone()
        matclone[:, :-1, :-1] = scale * rigidmat[:, :-1, :-1]

        transl = self.transl
        if self.around_center:
            # t' = t + c - (s * R) @ c
            # Critical: This calculation couples Scale and Translation. 
            # If Scale explodes, this term becomes massive, throwing the image out of FOV.
            # The clamping above protects this calculation.
            transl = transl + self.center - (matclone[:, :-1, :-1] @ self.center[..., None]).squeeze(-1)

        matclone[:, :-1, -1] = transl
        return matclone.contiguous() if homogenous else matclone[:, :-1, :].contiguous()
