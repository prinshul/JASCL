from models.encoder import Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch





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
    ):
        super().__init__(
            encoder_name=encoder_name,
            img_size=img_size,
            sub_norm=sub_norm,
            patch_size=patch_size,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
        )
        
        print('LinearDecoder')
        self.head = nn.Linear(self.embed_dim, num_classes)
        

    def forward(self, x: torch.Tensor, return_features=True) -> torch.Tensor:
        x = super().forward(x)
      
        feat = x.clone()
        logits = self.head(x)
        
        logits = logits.transpose(1, 2)
        logits = logits.reshape(logits.shape[0], -1, *self.grid_size)
        
        if return_features:
            feat = feat.transpose(1,2).reshape(feat.shape[0],-1,*self.grid_size)
            return logits, feat
        
        return logits
