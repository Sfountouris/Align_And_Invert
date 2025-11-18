# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from sklearn.decomposition import PCA
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct
import pywt
from sklearn.cluster import KMeans
def plot_wavelet_coeffs(tensor, name, save_path, wavelet='db1', level=None):
    """
    Applies wavelet transform to each row (patch) of a tensor and plots the coefficients.

    tensor: (1, 256, D)
    name: string identifier
    save_path: directory to save plot
    wavelet: type of wavelet (e.g., 'db1', 'haar', 'coif1')
    level: number of decomposition levels (optional)
    """
    os.makedirs(save_path, exist_ok=True)
    tensor = tensor.squeeze(0).cpu().numpy()  # shape: (256, D)

    coeff_matrix = []
    for patch in tensor:
        coeffs = pywt.wavedec(patch, wavelet=wavelet, level=level)
        coeffs_flat = np.concatenate(coeffs)
        coeff_matrix.append(coeffs_flat)

    coeff_matrix = np.array(coeff_matrix)  # shape: (256, variable)

    plt.figure(figsize=(10, 5))
    plt.imshow(np.log1p(np.abs(coeff_matrix)), aspect='auto', cmap='viridis')
    plt.colorbar(label="log(1 + |coeff|)")
    plt.xlabel("Wavelet Coefficient Index")
    plt.ylabel("Patch Index")
    plt.title(f"Wavelet Coefficients of {name}")
    plt.tight_layout()
    file_path = os.path.join(save_path, f"{name}_wavelet_coeffs.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved: {file_path}")

def plot_singular_values(tensor, name, save_path, k=32, energy_threshold=0.9):
    os.makedirs(save_path, exist_ok=True)

    tensor = tensor.squeeze(0)  # shape: (256, D)
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)

    # Compute cumulative energy ratio
    energy = S ** 2
    total_energy = energy.sum()
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy

    # Determine the effective rank for two energy thresholds
    rank_90 = (cumulative_energy >= 0.9).nonzero(as_tuple=True)[0][0].item() + 1
    rank_99 = (cumulative_energy >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1

    # Plot singular values
    plt.figure()
    plt.plot(S[:k].cpu().numpy(), marker='o', label="Singular Values")
    plt.axvline(rank_90 - 1, color='r', linestyle='--', label=f'90% energy at rank {rank_90}')
    plt.axvline(rank_99 - 1, color='b', linestyle='--', label=f'99% energy at rank {rank_99}')
    plt.title(f"Singular Values of {name}")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(save_path, f"{name}_singular_values.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved singular value plot: {file_path}")

    # Low-rank approximation using top-k singular values
    # S_low = torch.diag(S[:k])
    # U_low = U[:, :k]
    # Vh_low = Vh[:k, :]
    # approx = U_low @ S_low @ Vh_low  # shape: (256, D)

    # return approx.unsqueeze(0)


def plot_dct_coefficients(tensor, name, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    tensor = tensor.squeeze(0)  # shape: (256, D)
    data_np = tensor.cpu().numpy()

    dct_data = dct(data_np, axis=1, norm='ortho')

    plt.figure(figsize=(10, 4))
    plt.imshow(np.log1p(np.abs(dct_data)), aspect='auto', cmap='viridis')
    plt.colorbar(label='log(1 + |DCT|)')
    plt.title(f"DCT Coefficients of {name}")
    plt.xlabel("Frequency")
    plt.ylabel("Patch Index")
    plt.tight_layout()

    file_path = os.path.join(save_path, f"{name}_dct.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved DCT plot to: {file_path}")


def visualize_pca_embedding(tensor, name, save_path, upscale=16):
    os.makedirs(save_path, exist_ok=True)

    tensor = tensor.squeeze(0)  # shape: (256, D)
    data_np = tensor.cpu().numpy()

    # Apply PCA to reduce to 3 dimensions (for RGB)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_np)

    # Normalize to [0, 1]
    pca_min = pca_result.min(axis=0, keepdims=True)
    pca_max = pca_result.max(axis=0, keepdims=True)
    pca_normalized = (pca_result - pca_min) / (pca_max - pca_min)

    # Reshape to 16x16 grid of RGB pixels
    img = pca_normalized.reshape(16, 16, 3)

    # Upscale each pixel to a 16x16 block
    img_upsampled = np.kron(img, np.ones((upscale, upscale, 1)))

    # Convert to 8-bit RGB
    img_uint8 = (img_upsampled * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)

    file_path = os.path.join(save_path, f"{name}_pca_256x256.png")
    img_pil.save(file_path)
    print(f"Saved PCA upscaled visualization to: {file_path}")

def plot_segmentation_from_embeddings(tensor, name, save_path, patch_size=(16, 16), n_clusters=2):
    """
    Generates a segmentation map from patch embeddings using KMeans clustering.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, P, D), where P = H * W patches.
        name (str): Name for the saved file.
        save_path (str): Directory to save the output image.
        patch_size (tuple): Tuple (H, W) representing spatial layout of patches.
        n_clusters (int): Number of clusters for KMeans.
    """
    os.makedirs(save_path, exist_ok=True)

    tensor = tensor.squeeze(0)  # shape: (P, D)
    patch_embeddings = tensor.cpu().numpy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(patch_embeddings)
    labels = kmeans.labels_.reshape(patch_size)

    # Plot segmentation map
    plt.figure(figsize=(4, 4))
    plt.imshow(labels, cmap='tab10')
    plt.title(f"Segmentation from Embeddings: {name}")
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    file_path = os.path.join(save_path, f"{name}_segmentation.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved segmentation map to: {file_path}")


# class PatchConditionedProjector(nn.Module):
#     def __init__(self, in_channels=3, patch_size=16, latent_dim=768, rank=10):
#         super().__init__()
#         self.rank = rank
#         self.latent_dim = latent_dim

#         # CNN encoder to map image patch -> matrix
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),  # outputs B x 64 x 1 x 1
#             nn.Flatten(),             # B x 64
#             nn.Linear(64, latent_dim * rank)
#         )

#     def forward(self, h_t, x_patch):
#         """
#         h_t:       [B, D] latent vectors
#         x_patch:   [B, C, H, W] corresponding image patches
#         """
#         B, D = h_t.shape

#         # Predict U_x from image patch
#         U_flat = self.encoder(x_patch)              # [B, D*R]
#         U = U_flat.view(B, D, self.rank)            # [B, D, R]

#         # Project h_t using U U^T
#         U_T = U.transpose(1, 2)                     # [B, R, D]
#         projection = torch.bmm(U, torch.bmm(U_T, h_t.unsqueeze(2))).squeeze(2)  # [B, D]

#         return projection, U



def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, return_logvar=False, index = 0):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                if index % 10 == 0 or index == 249:
                    plot_singular_values(x, f'_iteration_{index}', 'svd_results_h')
                    plot_singular_values(zs[0], f'_iteration_{index}', 'svd_results_z')
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                   

        return x, zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

