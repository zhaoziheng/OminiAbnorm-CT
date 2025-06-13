import random
from typing import Tuple, Union, List
import math
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import torch 
from einops import rearrange, repeat, reduce
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from positional_encodings.torch_encodings import PositionalEncoding2D

from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer

from utils.distribute import main_print

class UNET(nn.Module):
    def __init__(self, model='unet', input_channels=9):
        """
        """
        super().__init__()
        
        self.input_channels = input_channels
        
        self.backbone = {
            'unet': PlainConvUNet(
                        input_channels=input_channels, 
                        n_stages=6, 
                        features_per_stage=(32, 64, 128, 256, 512, 1024), 
                        conv_op=nn.Conv2d, 
                        kernel_sizes=3, 
                        strides=(1, 2, 2, 2, 2, 2), 
                        n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                        n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                        conv_bias=True, 
                        norm_op=nn.InstanceNorm2d,
                        norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                        dropout_op=None,
                        dropout_op_kwargs=None,
                        nonlin=nn.LeakyReLU, 
                        nonlin_kwargs=None,
                        nonlin_first=False
                    ),
        }[model]
        
        self.avg_pool_ls = [    
            nn.AvgPool2d(32, 32),
            nn.AvgPool2d(16, 16),
            nn.AvgPool2d(8, 8),
            nn.AvgPool2d(4, 4),
            nn.AvgPool2d(2, 2),
            ]
        
        # 32 + 64 + 128 + 256 + 512 + 1024 --> 768
        self.projection_layer = nn.Sequential(
            nn.Linear(2016, 1536),
            nn.GELU(),
            nn.Linear(1536, 768),
            nn.GELU()
            )
        
        # positional encoding
        pos_embedding = PositionalEncoding2D(768)(torch.zeros(1, 16, 16, 768)) # b h/p w/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w c -> (h w) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=768, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(768)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        self.query_proj = nn.Sequential(
            nn.Linear(3584, 1536),
            nn.GELU(),
            nn.Linear(1536, 768),
            nn.GELU()
            )
        
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            )

        self.backbone.apply(InitWeights_He(1e-2))
        
    def load_saved(self, ckpt):
        # a ckpt saved from the segmentation training (only with segmentation model)
        if "model_state_dict" in ckpt:
            old_model_state_dict = ckpt["model_state_dict"]
            new_state_dict = {}
            for key, value in old_model_state_dict.items():
                if key.startswith('module.model.'): # if pretrained
                    new_key = key.replace('module.model.', 'backbone.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            # Check parameter alignment
            model_keys = set(self.state_dict().keys())
            checkpoint_keys = set(new_state_dict.keys())
            # 1. Missing parameters in the checkpoint
            missing_keys = model_keys - checkpoint_keys
            if missing_keys:
                main_print(f"** SEG MODEL ** Parameters not in checkpoint: {missing_keys}")
            # 2. Loaded parameters from the checkpoint
            loaded_keys = model_keys.intersection(checkpoint_keys)
            if loaded_keys:
                main_print(f"** SEG MODEL ** Parameters loaded from checkpoint: {loaded_keys}")
            self.load_state_dict(new_state_dict, strict=False)
        # a ckpt saved from the vlm-segmentation training
        else:
            state_dict = ckpt['seg_model']
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                  if 'transformer_decoder.layers' not in k or 'norm1' not in k}
            main_print(f"Skipped {len(state_dict) - len(filtered_state_dict)} parameters containing 'transformer_decoder.layers' and 'norm1'")
            self.load_state_dict(filtered_state_dict, strict=False)  # Using strict=False since we're deliberately skipping some parameters
        
    def frozen(self, freeze_type):
        # freeze_type: String or list specifying which parts to freeze
        for param in self.parameters():
            param.requires_grad = True
            
        # Default - no freezing
        if freeze_type is None or freeze_type == 'none':
            return
        freeze_parts = [freeze_type] if isinstance(freeze_type, str) else freeze_type
        
        # Handle different freezing options
        for part in freeze_parts:
            if part == 'all':
                for param in self.parameters():
                    param.requires_grad = False
                    
            elif part == 'backbone':
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
            elif part == 'transformer':
                for param in self.transformer_decoder.parameters():
                    param.requires_grad = False
                    
            elif part == 'vision_proj':
                for param in self.projection_layer.parameters():
                    param.requires_grad = False
                    
            elif part == 'query_proj':
                for param in self.query_proj.parameters():
                    param.requires_grad = False
                    
            elif part == 'mask_embed':
                for param in self.mask_embed_proj.parameters():
                    param.requires_grad = False

    def forward(self, image_input, queries):
        # image_input : (b d h w)
        # query_embedding : (b n dim)
        
        # Image Encoder and Pixel Decoder
        latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input) # B Dim H/P W/P
        
        # avg pooling each multiscale feature to H/P W/P
        image_embedding = []
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            tmp = avg_pool(latent_embedding)
            image_embedding.append(tmp)   # B ? H/P W/P D/P
        image_embedding.append(latent_embedding_ls[-1])
        
        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b dim h w -> b h w dim')
        
        image_embedding = self.projection_layer(image_embedding)    # B H/P W/P Dim
        image_embedding = rearrange(image_embedding, 'b h w dim -> (h w) b dim')    # (H/P W/P) B Dim
        
        # add pe to image embedding
        pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P) B Dim
        
        # query decoder
        B, N, _ = queries.shape
        
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim
        queries = self.query_proj(queries)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 32
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        logits = torch.einsum('bchw,bnc->bnhw', per_pixel_embedding, last_mask_embedding)
        
        return logits   # b, n, h, w
        
if __name__ == '__main__':
    model = UNET()  
    image = torch.rand((2, 9, 256, 256))
    segmentations = model(image)
    print(segmentations.shape)