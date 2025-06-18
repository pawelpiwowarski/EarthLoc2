import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicallyNeutralMinerLoss(nn.Module):
    """
    Soft‐margin “neutral” loss: for each (i,j),
      w * max(0, d_ij - m_pos)
    + (1-w) * max(0, m_neg - d_ij),
    where d_ij = 1 - cosine(emb_i, emb_j), w = Jaccard(i,j).
    """
    def __init__(self, margin_pos: float = 0.2, margin_neg: float = 0.8):
        super().__init__()
        self.m_pos, self.m_neg = margin_pos, margin_neg

    def forward(self,
                embeddings: torch.Tensor,   # [T, D], L2‐normalized
                jaccard:    torch.Tensor,   # [T, T], in [0,1]
                miner_outputs: tuple        # (anchors, negatives)
               ) -> torch.Tensor:
        anchors, negatives = miner_outputs   # each [N,]
        # cosine similarities
        sim = (embeddings[anchors] * embeddings[negatives]).sum(dim=1)
        d   = 1.0 - sim                       # distances in [0,2]
        w   = jaccard[anchors, negatives]    # weights in [0,1]

        # weighted soft‐margins
        loss = w * F.relu(d - self.m_pos) \
             + (1 - w) * F.relu(self.m_neg - d)
        return loss.mean()




class OrdinalTripletLossCosine(nn.Module):
    """
    Ordinal Triplet Loss using Cosine Distance for three classes: positive,
    neutral, and negative, relative to an anchor. Assumes embeddings are
    L2-normalized.

    This loss encourages the following ordering of cosine similarities
    with an anchor (A):
    sim(A, Positive) > sim(A, Neutral) > sim(A, Negative).

    It consists of two triplet-like components based on cosine similarity (sim):
    1. Positive vs. Neutral: Ensures sim(A,P) is greater than sim(A,Ntl) by a
       margin `m_pos`.
       Loss_PN = max(0, sim(A,Ntl) - sim(A,P) + m_pos)
    2. Neutral vs. Negative: Ensures sim(A,Ntl) is greater than sim(A,Neg) by a
       margin `m_neg`.
       Loss_NN = max(0, sim(A,Neg) - sim(A,Ntl) + m_neg)

    The total loss is the sum of these two components, averaged over the batch.
    """

    def __init__(self, margin_pos: float = 0.1, margin_neg: float = 0.1):
        """
        Initializes the OrdinalTripletLossCosine.

        Args:
            margin_pos (float): The margin for cosine similarity difference
                                between positive and neutral samples.
                                sim(A,P) should be > sim(A,Ntl) by at least
                                this margin. Default is 0.1.
            margin_neg (float): The margin for cosine similarity difference
                                between neutral and negative samples.
                                sim(A,Ntl) should be > sim(A,Neg) by at least
                                this margin. Default is 0.1.
        """
        super().__init__()
        self.m_pos = margin_pos
        self.m_neg = margin_neg

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        neutral_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Ordinal Triplet Loss using cosine similarity.
        Assumes embeddings are L2-normalized.

        Args:
            anchor_embeddings (torch.Tensor): L2-normalized embeddings of the
                                              anchor samples.
                                              Shape: (batch_size, embedding_dim).
            positive_embeddings (torch.Tensor): L2-normalized embeddings of the
                                                positive samples.
                                                Shape: (batch_size, embedding_dim).
            neutral_embeddings (torch.Tensor): L2-normalized embeddings of the
                                               neutral samples.
                                               Shape: (batch_size, embedding_dim).
            negative_embeddings (torch.Tensor): L2-normalized embeddings of the
                                                negative samples.
                                                Shape: (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The mean ordinal triplet loss for the batch (scalar).
        """

        # Calculate cosine similarities (dot product for normalized embeddings)
        # sim(anchor, positive)
        sim_ap = torch.sum(anchor_embeddings * positive_embeddings, dim=1)
        # sim(anchor, neutral)
        sim_an = torch.sum(anchor_embeddings * neutral_embeddings, dim=1)
        # sim(anchor, negative)
        sim_ane = torch.sum(anchor_embeddings * negative_embeddings, dim=1)

        # Loss component 1: Positive vs Neutral
        # We want sim(A,P) > sim(A,Ntl) + m_pos
        # This means sim(A,Ntl) - sim(A,P) + m_pos should be < 0 for no loss.
        # Loss_PN = max(0, sim(A,Ntl) - sim(A,P) + m_pos)
        loss_pos_neutral = F.relu(sim_an - sim_ap + self.m_pos)

        # Loss component 2: Neutral vs Negative
        # We want sim(A,Ntl) > sim(A,Neg) + m_neg
        # This means sim(A,Neg) - sim(A,Ntl) + m_neg should be < 0 for no loss.
        # Loss_NN = max(0, sim(A,Neg) - sim(A,Ntl) + m_neg)
        loss_neutral_neg = F.relu(sim_ane - sim_an + self.m_neg)

        # Total loss is the sum of the two components for each item in the batch
        total_loss_per_item = loss_pos_neutral + loss_neutral_neg

        # Return the mean loss over the batch
        return total_loss_per_item.mean()

