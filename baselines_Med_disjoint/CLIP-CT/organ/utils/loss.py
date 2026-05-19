import torch.nn as nn
import torch.nn.functional as F
import torch


class CosineKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', norm='L2'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm.upper()

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        if self.norm == "L2":
            loss = ((inputs - targets)**2).mean(dim=1)
        else:
            loss = (inputs - targets).mean(dim=1)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets / self.alpha, dim=1)

        loss = -(outputs * labels).mean(dim=1) * (self.alpha ** 2)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


# MiB Losses
class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = 1 - self.crit(x, y)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
        return - loss

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg
    
class DiceLoss1(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss1, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, organ_list):
        total_loss = []
        predict = F.sigmoid(predict)

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            for organ in range(organ_list):

                dice_loss = self.dice(predict[b, organ-1, :], target[b, organ-1, :].float())
                total_loss.append(dice_loss)
            
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]


class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, organ_list):
        # print(predict.shape, target.shape)
        # assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'
        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # target_mod = target[b, :].float().squeeze(0)
            target_mod = target[b, :].float().squeeze(0)
            # print(predict[b, :].shape, target_mod.shape)

            for organ in range(organ_list):
                # filtered_tensor = torch.where(target_mod == organ, target_mod, torch.zeros_like(target_mod))
                # ce_loss = self.criterion(predict[b, organ], filtered_tensor)
                # print(torch.unique(predict[b, organ, :]), torch.unique(target_mod[organ, :]))

                ce_loss = self.criterion(predict[b, organ, :], target_mod[organ, :])
                # print(ce_loss.item())
                total_loss.append(ce_loss)
                # if ce_loss.item() == 0:
                #     print(torch.unique(predict[b, organ, :]), torch.unique(target_mod[organ, :]))
                #     print(organ, ce_loss.item())
                # print(ce_loss.item())
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]