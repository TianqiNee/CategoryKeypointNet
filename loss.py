import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapLoss(nn.Module):
    """
    Heatmap loss function, calculates the loss between the predicted heatmap and the ground truth heatmap.
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

class KeypointLoss(nn.Module):
    # Traditional method loss for comparison
    def __init__(self):
        super(KeypointLoss, self).__init__()

    def forward(self, all_scores, gt_heatmap, keypoints_list):
        """
        Calculate the keypoint loss.
        scores: B*K*N, scores of the predicted candidate keypoints [0,1]
        gt_heatmap: B*N*H*W, mask of the ground truth keypoints, {0,1}
        keypoints: B*K*2, coordinates of the candidate keypoints
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_scores = all_scores.to(device)
        gt_heatmap = gt_heatmap.to(device)
        keypoints_list = keypoints_list.to(device)

        batch_size = gt_heatmap.shape[0]
        N = gt_heatmap.shape[1]
        K = all_scores.shape[1]

        loss = 0.0
        neg_loss = 0.0
        neg_count = 0
        pos_loss = 0.0

        for i in range(batch_size):
            keypoints = keypoints_list[i]  # [K, 2]
            scores = all_scores[i]         # [K, N]
            mask = gt_heatmap[i]          # [N, H, W]

            for n in range(N):
                gt = mask[n, :, :].nonzero().squeeze()  # Coordinate points of the nth keypoint
                if gt.dim() == 1:
                    gt = gt.unsqueeze(0)

                for k in range(K):
                    dists = torch.norm(keypoints[k] - gt, dim=1)  # Distance to all ground truth points
                    d = torch.min(dists)  # Take the closest distance as the judgment basis

                    if d < 1.0:  # Assume that points less than 1 pixel are positive samples
                        pos_loss += 10000 / (1 + torch.exp(d)) * torch.log(scores[k, n])
                    else:
                        neg_loss += torch.log(1 - scores[k, n])
                        neg_count += 1

        # Add the positive sample loss directly
        loss -= pos_loss

        # Normalize and add the negative sample loss
        if neg_count > 0:
            loss -= 10000 / neg_count * neg_loss
        return loss