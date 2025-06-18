import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from types import SimpleNamespace as Namespace

# Assuming these are in your project structure
from apl_models.salad import SALAD
from apl_models.mixvpr import MixVPR




class DINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        image_size=518,  # Default for DINOv2 models
        model_type="vit_base_patch14_reg4_dinov2.lvd142m",
        num_of_layers_to_unfreeze=1,
        desc_dim=768,  # vit-base has 768-dim embeddings
        aggregator_type="No",
    ):
        super().__init__()

        # Initialize backbone with registers
        self.backbone = timm.create_model(
            model_type, pretrained=True, num_classes=0, img_size=image_size
        )

        # Store configuration parameters
        self.model_type = model_type
        self.num_channels = self.backbone.embed_dim
        self.desc_dim = desc_dim
        self.image_size = image_size
        self.num_of_layers_to_unfreeze = num_of_layers_to_unfreeze
        self.aggregator_type = aggregator_type
        self.aggregator = None

        if aggregator_type == "SALAD":
            if "vit_small" in model_type:
                self.aggregator = SALAD(
                    num_channels=self.num_channels,
                    num_clusters=24,
                    cluster_dim=64,
                    token_dim=512,
                    dropout=0.3,
                )
                # Output: 512 + (24 * 64) = 2,048 dims
                self.desc_dim = 512 + (24 * 64)
            elif "vit_base" in model_type:
                self.aggregator = SALAD(
                    num_channels=self.num_channels,
                    num_clusters=32,
                    cluster_dim=64,
                    token_dim=1024,
                    dropout=0.3,
                )
                # Output: 1024 + (32 * 64) = 3,072 dims
                self.desc_dim = 1024 + (32 * 64)
            elif "vit_large" in model_type:
                self.aggregator = SALAD(
                    num_channels=self.num_channels,
                    num_clusters=48,
                    cluster_dim=64,
                    token_dim=1024,
                    dropout=0.3,
                )
                # Output: 1024 + (48 * 64) = 4,096 dims
                self.desc_dim = 1024 + (48 * 64)
        elif aggregator_type == "MixVPR":
            patch_dim = image_size // 14
            if "vit_small" in model_type:
                out_dim = 2048
            elif "vit_base" in model_type:
                out_dim = 3072
            elif "vit_large" in model_type:
                out_dim = 4096
            else:
                # Default or error
                out_dim = 4096

            self.aggregator = MixVPR(
                in_channels=self.num_channels,
                in_h=patch_dim,
                in_w=patch_dim,
                out_channels=out_dim,
            )
            self.desc_dim = out_dim
        else:
            print("Warning: no aggregator chosen, if this is what you meant ignore this")

        # This should be called regardless of the aggregator type.
        self._freeze_parameters()

    def _freeze_parameters(self):
        """
        Freeze all parameters except the last N transformer blocks and norm layer.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        if self.num_of_layers_to_unfreeze > 0:
            for block in self.backbone.blocks[
                -self.num_of_layers_to_unfreeze :
            ]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze norm layer
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # Count trainable parameters for backbone
        def count_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logging.info(
            f"Number of trainable parameters backbone: {count_trainable_params(self.backbone):,}"
        )

        # Count aggregator parameters if it exists
        if self.aggregator is not None:
            aggregator_params = count_trainable_params(self.aggregator)
            logging.info(
                f"Number of trainable parameters aggregator: {aggregator_params:,}"
            )
            logging.info(
                f"Total trainable parameters: {count_trainable_params(self.backbone) + aggregator_params:,}"
            )

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.backbone.forward_features(x)

        # Consistent handling for register vs. non-register models
        if self.aggregator_type in ["SALAD", "MixVPR"]:
            # DINOv2 with registers has 4 register tokens + 1 CLS token
            # Standard ViT has 1 CLS token
            start_index = 5 if "reg" in self.model_type else 1
            patch_tokens = x[:, start_index:]

            # Reshape to (B, C, H, W) for aggregators
            patch_tokens_map = patch_tokens.reshape(
                (B, H // 14, W // 14, self.num_channels)
            ).permute(0, 3, 1, 2)

            if self.aggregator_type == "SALAD":
                cls_token = x[:, 0]
                return self.aggregator((patch_tokens_map, cls_token))
            elif self.aggregator_type == "MixVPR":
                return self.aggregator(patch_tokens_map)

        # Default behavior: extract features from CLS pooling
        features = self.backbone.forward_head(x, pre_logits=True)

        # L2 normalization
        return F.normalize(features, p=2, dim=-1)