import torch
import torch.nn as nn
#import wandb
import os, sys, random, pickle
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from training.lightning_module import LightningModule

def norm_mean(x):
    # x should be N x F, return 1 x F
    return F.normalize(x, dim=1).mean(dim=0, keepdim=True)




class LinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
        )

        self.save_hyperparameters()

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        
        self.num_classes=num_classes
        self.protos_dict = {}
        self.protos = {}
        self.bkg_proto = {}
        
        for cl in range(self.num_classes):
            #self.protos_dict[cl]=[]
            self.protos[cl]=[]
            self.bkg_proto[cl]=[]
        
        self.prev_protos=None

        if self.num_classes>10:
            print("Previous prototypes loaded.")
            if self.num_classes==15:
                print('full_protos_base.pkl')
                with open('full_protos_base.pkl', 'rb') as f:
                    self.prev_protos=pickle.load(f)
            else:
                print('full_protos_base.pkl')
                with open('full_protos_base.pkl', 'rb') as f:
                    self.prev_protos=pickle.load(f)
        
        
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        
        L_proto=0
        # for label, feat in self.prev_protos.items():
        #     if type(label)==str:
        #         label=0    
        #     #feat=feat.unsqueeze(0)
        #     #print("protos shape ",label.shape,feat.shape)
        #     label=torch.tensor([label])
        #     label=label.view(1)
        #     label=label.cuda()
            
        #     feat=feat.squeeze(-1).squeeze(-1)
        #     feat=feat.cuda()
            
        #     #print("Proto match \n",feat.shape, label.shape)
            
        #     logits=self.network.head(feat)
            
        #     #print(logits.shape, label.shape)
        #     loss_p_ce=self.criterion(logits, label.long())
        #     L_proto+=loss_p_ce
        
        
        #for keys in imgs:
        #    print(keys.shape)
        ## img shape - ([3, 1024, 1024])  
        
        #for keys in targets:
        #    print(type(keys)) str
        #    for k in keys:
        #        print(k,keys[k].shape)
        #    masks torch.Size([3, 1024, 1024])
        #   labels torch.Size([3])
        #print(len(imgs),len(targets))
        #print(imgs[0].shape, targets[0].shape)
        #print(imgs.shape, targets.shape)
        
        logits, feat = self(imgs)
        
        '''
        with torch.no_grad():
            grad_update = (self.network.head.mu.weight.grad.clone().detach())**2
            #max_val=10.0
            #grad_update = torch.clamp(grad_update,min=0, max=max_val)
            self.network.head.grad_update.data = grad_update
            del grad_update
        '''
        #print(self.network(imgs).shape)
        #print(self.network.encoder(imgs).shape)
        ## print("New ",logits.shape) torch.Size([1, 10, 64, 64])
        #print("linear semantic logits, feat ",logits.shape, feat.shape)
        ##linear semantic logits, feat  torch.Size([1, 15, 64, 64]) torch.Size([1, 768, 64, 64])
        
        logits = F.interpolate(logits, self.img_size, mode="bilinear")
        #print(logits[0].shape, targets[0].shape)
        ## print("Interpolate logits",logits.shape) torch.Size([1, 10, 1024, 1024])
        
        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()
        
        loss_total = self.criterion(logits, targets) + 0.2*L_proto
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        #print("Stack targets", targets.shape)
        return loss_total
    
    
    def create_protoypes(self, logits, feats, targets):
        #print("create_protoypes ",logits.shape, feats.shape, targets.shape)
        #create_protoypes  torch.Size([1, 10, 1024, 1024]) torch.Size([1, 768, 64, 64]) torch.Size([1, 1024, 1024])
        
        
        for cl in range(self.num_classes):
    
            feat_flat = feats.squeeze(0).view(768, -1)
            labels_down = F.interpolate(targets.float(), size=feats.shape[-2:], mode='bilinear', align_corners=False).long()  
            
            mask = (labels_down == cl)          # [128, 256]
            if mask.sum() == 0:
                continue
                
            mask_flat = mask.view(-1)
            #print(mask_flat.shape)
            
            
            cls_feats = feat_flat[:, mask_flat]           # [304, N]
            cls_feats = cls_feats.transpose(0, 1)   # [N, 304]
            #print(cls_feats.shape)
            protos_curr=norm_mean(cls_feats)
            bkg_proto_curr=norm_mean(feat_flat[:, ~mask_flat].transpose(0, 1))  
            #print(protos[0].shape)
            #del inputs, targets, feats, outputs
            
            self.protos[cl].append(protos_curr)
            self.bkg_proto[cl].append(bkg_proto_curr)
        #self.protos_dict[cl].append([protos,bkg_proto])
        
        
    
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, targets = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits, fts = self(crops)
        
        
        #print(crop_logits.shape, fts.shape) torch.Size([10, 64, 64]) torch.Size([768, 64, 64])
        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        if is_notebook:
            return logits

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        #self.create_protoypes(crop_logits, fts, torch.stack(targets).long().unsqueeze(1))
        
        self.update_metrics(logits, targets, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(
                imgs[0],
                targets[0],
                logits=logits[0],
            )
            #self.trainer.logger.experiment.log({name: [wandb.Image(plot)]})  # type: ignore

    def on_validation_epoch_end(self):
        
        for cl in range(self.num_classes):
            #protos, bkg_proto = self.protos_dict[cl][0], self.protos_dict[cl][1]
            
            if len(self.protos[cl]) > 0:
                protos = torch.cat(self.protos[cl], dim=0)
                bkg_proto = torch.cat(self.bkg_proto[cl], dim=0)
                
                protos = protos.mean(dim=0)
                bkg_proto = bkg_proto.mean(dim=0)
                
                protos=protos.view(1, -1, 1, 1)
                bkg_proto=bkg_proto.view(1, -1, 1, 1)
                    
                #print("after view ",protos.shape, bkg_proto.shape)
                
                if protos is not None:
                    #wc = wc.unsqueeze(3)
                    self.protos_dict[cl]=protos
                    bgkey='bg'+str(cl)
                    self.protos_dict[bgkey]=bkg_proto
            
        for key, val in self.protos_dict.items():
            print(key, val.shape)
    
        #with open('full_protos_base.pkl', 'wb') as f:
            #pickle.dump(self.protos_dict, f)
        
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
