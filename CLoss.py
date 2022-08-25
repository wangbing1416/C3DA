import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, margin=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, source, positive, negative):

        source = F.normalize(source, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        sim_pos = F.cosine_similarity(source, positive)
        sim_neg = F.cosine_similarity(source, negative)
        loss = torch.sum(F.relu(sim_neg - sim_pos + self.margin))

        return 1.0 / self.batch_size * loss
