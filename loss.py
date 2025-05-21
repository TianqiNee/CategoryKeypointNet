import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapLoss(nn.Module):
    """
    Balanced focal loss, calculates the loss between the predicted heatmap and the ground truth heatmap.
    """
    def __init__(self, loss_type="mse"):
        """
        Initialize the heatmap loss function.
        :param loss_type: Type of loss function, supports "mse" or "focal".
        """
        super(HeatmapLoss, self).__init__()
        self.loss_type = loss_type
        self.step = 0

    def forward(self, pred_heatmap, gt_heatmap, mask, writer=None):
        """
        Calculate the heatmap loss.
        :param pred_heatmap: Predicted heatmap, shape (B, C, H, W).
        :param gt_heatmap: Ground truth heatmap, shape (B, C, H, W).
        :return: Loss value.
        """
        if self.loss_type == "mse":
            # Mean squared error loss
            gt_heatmap = gt_heatmap.float()
            mask1 = mask.repeat(1, gt_heatmap.shape[1], 1, 1)
            loss1 = F.mse_loss(pred_heatmap[mask1], gt_heatmap[mask1], reduction="mean")
            mask2 = (~mask).repeat(1, gt_heatmap.shape[1], 1, 1)   
            loss2 = F.mse_loss(pred_heatmap[mask2], gt_heatmap[mask2], reduction="mean")
            if writer is not None:
                writer.add_scalar('Loss/loss1', loss1.item(), self.step)
                writer.add_scalar('Loss/loss2', loss2.item(), self.step)
                self.step += 1
            loss = loss1 + loss2
            # loss = F.mse_loss(pred_heatmap, gt_heatmap, reduction="mean")
        elif self.loss_type == "focal":
            # Focal Loss, suitable for cases where positive samples in the heatmap are sparse
            pos_weights = gt_heatmap.eq(1.).float()
            med_weights = ((gt_heatmap != 0) & (gt_heatmap != 1)).float()
            neg_weights1 = mask.float() * (1 - pos_weights) * (1 - med_weights)
            neg_weights2 = gt_heatmap.eq(0.).float() * (1 - neg_weights1)

            pos_loss = -pos_weights * torch.log(pred_heatmap + 1e-6) * torch.pow(1 - pred_heatmap, 2)
            pos_loss = pos_loss.sum() / pos_weights.sum()

            med_loss = -med_weights * torch.log(1 - abs(gt_heatmap - pred_heatmap) + 1e-6) * torch.pow(1 - abs(gt_heatmap - pred_heatmap), 2)
            med_loss = med_loss.sum() / med_weights.sum()

            neg_loss1 = -neg_weights1 * torch.log(1 - pred_heatmap + 1e-6) * torch.pow(pred_heatmap, 2)
            neg_loss1 = neg_loss1.sum() / neg_weights1.sum()

            neg_loss2 = -neg_weights2 * torch.log(1 - pred_heatmap + 1e-6) * torch.pow(pred_heatmap, 2)
            neg_loss2 = neg_loss2.sum() / neg_weights2.sum()

            if writer is not None:
                writer.add_scalar('Loss/pos_loss', pos_loss.item(), self.step)
                writer.add_scalar('Loss/med_loss', med_loss.item(), self.step)
                writer.add_scalar('Loss/neg_loss1', neg_loss1.item(), self.step)
                writer.add_scalar('Loss/neg_loss2', neg_loss2.item(), self.step)
                self.step += 1
            loss = pos_loss + med_loss + neg_loss1 + neg_loss2
        else:
            raise ValueError("Unsupported loss type. Use 'mse' or 'focal'.")

        return loss

class CrossEntropyLoss(nn.Module):
    # Cross-entropy loss
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_heatmap, gt_heatmap):
        """
        Calculate the cross-entropy loss.
        :param pred_heatmap: Predicted heatmap, shape (B, C, H, W), probability values [0,1].
        :param gt_heatmap: Ground truth heatmap, shape (B, C, H, W), mask form 0/1.
        :C: Number of classes
        :return: Loss value.
        L_ce = ∑c_i*log2(p_i)
        """
        loss = -gt_heatmap * torch.log2(pred_heatmap + 10e-5)
        loss = loss.mean()
        return loss


        # Normalize and add the negative sample loss
        if neg_count > 0:
            loss -= 10000 / neg_count * neg_loss
        return loss
