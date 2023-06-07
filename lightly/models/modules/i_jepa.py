import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import partial
from typing import Optional, Callable, List
import math


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,

    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class IJEPA_Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=depth)
    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        #return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]


class IJEPA_Encoder(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        depth, 
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, src):
        return self.transformer_encoder(src)


class IJEPA_base(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, enc_depth, pred_depth, num_heads, post_emb_norm=False, M = 4, mode="train", layer_dropout=0.):
        super().__init__()
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        #define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_dim  = (self.patch_embed.patch_shape[0], self.patch_embed.patch_shape[1])
        self.num_tokens = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        #define the cls and mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = IJEPA_Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth, 
        )  
        self.student_encoder = copy.deepcopy(self.teacher_encoder).cuda()
        self.predictor = IJEPA_Encoder(embed_dim, num_heads, pred_depth)

    @torch.no_grad() 
    def get_target_block(self, target_encoder, x, patch_dim, aspect_ratio, scale, M):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        #get the patch dimensions
        patch_h, patch_w = patch_dim
        #get the number of patches
        num_patches = patch_h * patch_w
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            #get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            #get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)
                    
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.cuda(), target_patches, all_patches

    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)
        return x[:, patches, :]


    def forward(self, x, target_aspect_ratio=1, target_scale=1, context_aspect_ratio=1, context_scale=1):
        #get the patch embeddings
        x = self.patch_embed(x)
        b, n, e = x.shape
        x = x + self.pos_embedding[:, :n]
        #add the positional embeddings
        x = x + self.pos_embedding
        #normalize the embeddings
        x = self.post_emb_norm(x)
        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            return self.student_encoder(x)
        target_blocks, target_patches, all_patches = self.get_target_block(self.teacher_encoder, x, self.patch_dim, target_aspect_ratio, target_scale, self.M)
        m, b, n, e = target_blocks.shape

        context_block = self.get_context_block(x, self.patch_dim, context_aspect_ratio, context_scale, all_patches)
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)


        prediction_blocks = torch.zeros((m, b, n, e)).cuda()
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.pos_embedding[:, target_patches[i], :]
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks