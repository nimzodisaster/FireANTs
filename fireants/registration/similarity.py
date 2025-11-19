# fireants/registration/similarity.py

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
    for scaling (s_x = s_y = s_z), guaranteeing no shear and uniform growth.
    """

    def __init__(self, scales: List[float], iterations: List[int], 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                loss_type: str = "cc",
                optimizer: str = 'Adam', optimizer_params: dict = {},
                optimizer_lr: float = 3e-2,
                **kwargs
                ) -> None:
        
        # 1. Initialize Parent with scaling=False
        # We turn off the parent's anisotropic scaling because we want to define our own
        kwargs['scaling'] = False
        super().__init__(scales, iterations, fixed_images, moving_images, 
                         loss_type=loss_type, optimizer=optimizer, 
                         optimizer_params=optimizer_params, optimizer_lr=optimizer_lr,
                         **kwargs)
        
        # 2. Define Isotropic Scale Parameter [N, 1]
        # Log-scale: 0.0 means scale of 1.0
        device = fixed_images.device
        self.logscale = nn.Parameter(torch.zeros((self.opt_size, 1), device=device, dtype=self.dtype))

        # 3. Re-initialize Optimizer
        # The parent __init__ created an optimizer with only [rotation, transl].
        # We need to create a new one that includes our new self.logscale.
        params = [self.rotation, self.transl, self.logscale]
        
        if optimizer == 'SGD':
            self.optimizer = SGD(params, lr=optimizer_lr, **optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam(params, lr=optimizer_lr, **optimizer_params)
        
        logger.info("Initialized Similarity Registration (Rigid + Isotropic Scaling)")

    def get_rigid_matrix(self, homogenous=True):
        """
        Compute the transformation matrix using isotropic scaling.
        """
        rigidmat = self.get_rotation_matrix() # [N, dim+1, dim+1]
        
        # Expand scalar scale [N, 1] -> [N, 1, 1] for broadcasting
        scale = torch.exp(self.logscale)[..., None]  
        
        matclone = rigidmat.clone()
        # Broadcasting applies scale to all spatial dimensions uniformly
        matclone[:, :-1, :-1] = scale * rigidmat[:, :-1, :-1]
        
        transl = self.transl
        if self.around_center:
            # Recalculate center offset with the scaled rotation
            # t' = t + c - (s * R) @ c
            transl = transl + self.center - (matclone[:, :-1, :-1] @ self.center[..., None]).squeeze(-1)
            
        matclone[:, :-1, -1] = transl
        return matclone.contiguous() if homogenous else matclone[:, :-1, :].contiguous()
