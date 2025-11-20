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
    
    Inherits from RigidRegistration but enforces a single scalar learning parameter
    for scaling (s_x = s_y = s_z).
    
    Args:
        scale_only (bool): If True, locks Rotation and Translation (using values from initialization)
                           and optimizes ONLY the scale. Default: False.
    """

    def __init__(self, scales: List[float], iterations: List[int], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 3e-2,
                scale_only: bool = False,  # <--- NEW PARAMETER
                **kwargs
                ) -> None:
        
        # 1. Initialize Parent
        kwargs['scaling'] = False
        super().__init__(scales, iterations, fixed_images, moving_images, 
                         loss_type=loss_type, optimizer=optimizer, 
                         optimizer_params=optimizer_params, optimizer_lr=optimizer_lr,
                         **kwargs)
        
        # 2. Define Isotropic Scale Parameter
        device = fixed_images.device
        self.logscale = nn.Parameter(torch.zeros((self.opt_size, 1), device=device, dtype=self.dtype))

        # 3. Re-initialize Optimizer
        # If scale_only is True, we ONLY pass self.logscale to the optimizer.
        # self.rotation and self.transl remain as Parameters (so gradients are calculated),
        # but the optimizer ignores them, effectively freezing them at their initial values.
        if scale_only:
            params = [self.logscale]
            logger.info("Similarity Registration: Optimizing SCALE ONLY (Rotation/Translation frozen)")
        else:
            params = [self.rotation, self.transl, self.logscale]
            logger.info("Similarity Registration: Optimizing Scale, Rotation, and Translation")
        
        if optimizer == 'SGD':
            self.optimizer = SGD(params, lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam(params, lr=optimizer_lr, **optimizer_params)

    def get_rigid_matrix(self, homogenous=True):
        """
        Compute the transformation matrix using isotropic scaling.
        """
        rigidmat = self.get_rotation_matrix() # [N, dim+1, dim+1]
        scale = torch.exp(self.logscale)[..., None]  
        matclone = rigidmat.clone()
        matclone[:, :-1, :-1] = scale * rigidmat[:, :-1, :-1]
        
        transl = self.transl
        if self.around_center:
            # t' = t + c - (s * R) @ c
            transl = transl + self.center - (matclone[:, :-1, :-1] @ self.center[..., None]).squeeze(-1)
            
        matclone[:, :-1, -1] = transl
        return matclone.contiguous() if homogenous else matclone[:, :-1, :].contiguous()
