from models.encoder import Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class LinearDecoder(Encoder):
    def __init__(
        self,
        encoder_name,
        num_classes,
        img_size,
        sub_norm=False,
        patch_size=16,
        pretrained=True,
        ckpt_path="",
        temperature: float = 10.0,
    ):
        super().__init__(
            encoder_name=encoder_name,
            img_size=img_size,
            sub_norm=sub_norm,
            patch_size=patch_size,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
        )
        
        
        self.head_weight = nn.Parameter(torch.empty(num_classes, self.embed_dim))
        nn.init.kaiming_uniform_(self.head_weight, a=math.sqrt(5))

        self.register_buffer('inverse_grad', torch.zeros_like(self.head_weight))

        self.temp = temperature
        self.epsilon = 1e-8

        if self.head_weight.requires_grad:
            self.head_weight.register_hook(self._update_inverse_grad)
    
    def _update_inverse_grad(self, grad: torch.Tensor):
        with torch.no_grad():
            self.inverse_grad.copy_(grad.pow(2))
    
    def project(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.head_weight)

    def forward(self, x: torch.Tensor, return_features=True) -> torch.Tensor:
        x = super().forward(x)
      
        base_weight = self.head_weight
        inv_grad = self.inverse_grad
        
        inv_grad = 1 / (inv_grad + self.epsilon)
        min_val, max_val = inv_grad.min(), inv_grad.max()
        norm_inv_grad = (1 + inv_grad - min_val) / (1 + max_val - min_val + self.epsilon)
        
        noise = torch.randn_like(base_weight)
        perturbed_weight = base_weight + 1e-5*norm_inv_grad * noise
        print("perturbed")
	        
        norm_weight = F.normalize(perturbed_weight, p=2, dim=-1)
        norm_x = F.normalize(x, p=2, dim=-1)
        
        logits = F.linear(norm_x, norm_weight)
        
        logits = logits * self.temp
        
        logits = logits.transpose(1, 2)
        logits = logits.reshape(logits.shape[0], -1, *self.grid_size)
        
        if return_features:
            feat = x.transpose(1,2).reshape(x.shape[0],-1,*self.grid_size)
            return logits, feat
        
        return logits
