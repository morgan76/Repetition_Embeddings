import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, symmetrical=False):
        """
        Contrastive Loss with temperature scaling and optional symmetry, supporting multiple self-similarity matrices.

        Args:
            temperature (float): Scaling factor for the similarity scores.
            symmetrical (bool): Whether to compute the symmetrical version of the loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.symmetrical = symmetrical
    
    def forward(self, ssms_list, anchors, positives, negatives, embeddings):
        """
        Compute the (optionally symmetrical) contrastive loss for a batch using a list of self-similarity matrices.

        Args:
            ssms_list (list of torch.Tensor): List of self-similarity matrices.
            cs (torch.Tensor): Tensor indicating which ssm in `ssms_list` to use for each anchor.
            anchors (torch.Tensor): Indices of anchor samples, shape (n_anchors,).
            positives (torch.Tensor): Indices of positive samples, shape (n_anchors, n_positives).
            negatives (torch.Tensor): Indices of negative samples, shape (n_anchors, n_negatives).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        total_loss = 0.0
        batch_size = anchors.size(0)
        cs = torch.zeros(batch_size)
        
        for i, ssm in enumerate(ssms_list):
            # Mask for the current ssm
            mask = (cs == i)
            if not mask.any():
                continue

            # Extract indices for the current ssm
            anchors_i = anchors#[mask]
            positives_i = positives#[mask]
            negatives_i = negatives#[mask]

            # Scale the similarity matrix with temperature
            scaled_ssm = ssm / self.temperature

            # ** Forward Direction: Anchors -> Positives **
            # Extract positive and negative scores
            pos_scores = scaled_ssm[anchors_i.unsqueeze(1), positives_i]  # Shape: (n_anchors_i, n_positives)
            
            pos_scores_exp = torch.exp(pos_scores)

            neg_scores = scaled_ssm[anchors_i.unsqueeze(1), negatives_i]  # Shape: (n_anchors_i, n_negatives)
            neg_scores_exp = torch.exp(neg_scores).sum(dim=1, keepdim=True)  # Sum over negatives (n_anchors_i, 1)

            # Normalize positive scores and compute forward loss
            normalized_pos_scores = pos_scores_exp / (pos_scores_exp + neg_scores_exp + 1e-10)
            loss_forward = -torch.log(normalized_pos_scores + 1e-10).mean()

            if not self.symmetrical:
                total_loss += loss_forward
                continue

            # ** Reverse Direction: Positives -> Anchors **
            # Reshape positives and anchors for reverse calculation
            n_anchors_i, n_positives = positives_i.size()
            expanded_positives = positives_i.view(-1)  # Flatten positives
            expanded_negatives = negatives_i.unsqueeze(1).expand(n_anchors_i, n_positives, negatives_i.size(1)).contiguous()
            expanded_negatives = expanded_negatives.view(-1, negatives_i.size(1))

            pos_scores_reverse = scaled_ssm[expanded_positives, anchors_i.repeat_interleave(n_positives)]
            pos_scores_reverse_exp = torch.exp(pos_scores_reverse)

            neg_scores_reverse = scaled_ssm[expanded_positives.unsqueeze(1).expand(-1, negatives_i.size(1)), expanded_negatives]
            neg_scores_reverse_exp = torch.exp(neg_scores_reverse).sum(dim=1)

            # Reshape reverse scores and compute reverse loss
            pos_scores_reverse_exp = pos_scores_reverse_exp.view(n_anchors_i, n_positives)
            neg_scores_reverse_exp = neg_scores_reverse_exp.view(n_anchors_i, n_positives)

            normalized_pos_scores_reverse = pos_scores_reverse_exp / (pos_scores_reverse_exp + neg_scores_reverse_exp + 1e-10)
            loss_reverse = -torch.log(normalized_pos_scores_reverse + 1e-10).mean()

            # Add combined loss for this ssm
            total_loss += (loss_forward + loss_reverse) / 2

        return total_loss / len(ssms_list)
