import torch

class OrdinalQuadrupletMiner:
    """
    Mines multiple quadruplets per anchor using vectorized operations.
    Set num_samples to control how many (P, Ntl, Neg) triplets are generated per anchor.
    """

    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples


    def mine(self, descriptors, labels, is_overlapping):
        batch_size = descriptors.size(0)
        device = descriptors.device

        # ===== 1. Input Validation =====
        if batch_size == 0:
            return self._empty_result(device)

        # ===== 2. Precompute Key Masks =====
        labels = labels.view(-1)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_ne = ~labels_eq
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device= device)
        not_overlapping = ~is_overlapping

        # ===== 3. Find Valid Anchors =====
        valid_anchor_mask = (
            (labels_eq & diag_mask).any(1) &  # Has positives
            (labels_ne & is_overlapping).any(1) &               # Has neutrals
            (labels_ne & not_overlapping).any(1)                # Has negatives
        )
        valid_anchors = torch.where(valid_anchor_mask)[0]

        if valid_anchors.numel() == 0:
            return self._empty_result(device)

        # ===== 4. Batch-Aware Sampling =====
        def sample_multiple(mask, num_samples):
            """Samples multiple indices per anchor from mask"""
            counts = mask.sum(dim=1)
            max_samples = min(num_samples, counts.max().item()) if counts.numel() > 0 else 0
            
            # Generate random indices using matrix operations
            rand = torch.rand(len(valid_anchors), max_samples, device=device )

            cum_probs = mask[valid_anchors].float().cumsum(dim=1)
            scaled = (rand * cum_probs[:, -1:]).long()
            samples = torch.searchsorted(cum_probs, scaled)
            
            # Mask invalid samples
            valid = samples < counts[:, None]
            return samples, valid

        # Sample multiple candidates per anchor type
        pos_samples, pos_valid = self._sample_from_mask(
            labels_eq & diag_mask , valid_anchors
        )
        ntl_samples, ntl_valid = self._sample_from_mask(
            labels_ne & is_overlapping, valid_anchors
        )
        neg_samples, neg_valid = self._sample_from_mask(
            labels_ne & not_overlapping, valid_anchors
        )

        # ===== 5. Generate Expanded Anchors =====
        # Repeat anchors for each sample and filter valid combinations
        anchors_expanded = valid_anchors.repeat_interleave(self.num_samples)
        valid_mask = (pos_valid & ntl_valid & neg_valid).flatten()
    
        
        return (
            anchors_expanded[valid_mask],
            pos_samples.flatten()[valid_mask.flatten()],
            ntl_samples.flatten()[valid_mask.flatten()],
            neg_samples.flatten()[valid_mask.flatten()],
        )

    def _sample_from_mask(self, full_mask, valid_anchors, num_samples=None):
        num_samples = num_samples or self.num_samples
        mask = full_mask[valid_anchors]
        
        # Vectorized multinomial sampling
        counts = mask.sum(dim=1)
        probs = mask.float() / counts[:, None].clamp(min=1)
        samples = torch.multinomial(
            probs, 
            num_samples=num_samples, 
            replacement=True, 
        )
        
        # Mark valid samples (those that didn't use replacement)
        valid = torch.arange(mask.size(1), device=mask.device)[None] < counts[:, None]
        return samples, valid[:, :num_samples]

    def _empty_result(self, device):
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )