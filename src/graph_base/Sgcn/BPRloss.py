import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, y_pred, y_true):
        pos_pred = torch.masked_select(y_pred, y_true.bool())
        neg_pred = torch.masked_select(y_pred, ~y_true.bool())     
        sampling = neg_pred[torch.randperm(neg_pred.size(0))[:pos_pred.size(0)-neg_pred.size(0)]]    
        neg_pred = torch.concat((neg_pred,sampling))
        indices = torch.randperm(len(neg_pred))
        neg_pred = neg_pred[indices]
        #print(pos_pred[torch.randperm(pos_pred.size(0))[:5]])
        #print()\
        diff = pos_pred - neg_pred
        loss = -torch.sum(torch.log(torch.sigmoid(diff)))
        return loss