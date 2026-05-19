from models.encoder import Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch


class StochasticClassifier(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        self.mu = nn.Linear(num_features, num_classes,bias=False)
        self.sigma = nn.Linear(num_features, num_classes, bias=False)
        self.temp = 4
        self._init_weights()
        print('Linear Stochastic Classifier')
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.modules.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
    
    def forward(self, x, stochastic=True):
        mu = self.mu.weight
        sigma = self.sigma.weight
        
        if stochastic:
            sigma = F.softplus(sigma - 4) 
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        
        weight = F.normalize(weight, p=2, dim=0)
        x = F.normalize(x, p=2, dim=0)

        score = F.linear(x, weight)
        score = score*self.temp

        return score





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
        #self.head = StochasticClassifier(self.embed_dim, num_classes)
        

    def forward(self, x: torch.Tensor, return_features=True) -> torch.Tensor:
        x = super().forward(x)
        # print("linear decoder super().forward(x)",x.shape)
        # linear decoder super().forward(x) torch.Size([1, 4096, 768])
        feat = x.clone()
        logits = self.head(x)
        #print("linear decoder self.head(x)",x.shape)
        #linear decoder self.head(x) torch.Size([1, 4096, 768])
        logits = logits.transpose(1, 2)
        logits = logits.reshape(logits.shape[0], -1, *self.grid_size)
        
        if return_features:
            feat = feat.transpose(1,2).reshape(feat.shape[0],-1,*self.grid_size)
            return logits, feat
        
        return logits
