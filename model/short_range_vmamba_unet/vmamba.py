import math
from functools import partial
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_

from torchvision import transforms
import torchvision.models as models
# 导入imagefolder
import torch.nn.functional as F
from PIL import Image
import os

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

#################################################计算相似度矩阵传递给A进行初始化#########################################
class ResNetFeatures(nn.Module):
    def __init__(self, weights_path):
        super(ResNetFeatures, self).__init__()
        resnet = models.resnet50(weights=None)
        resnet.load_state_dict(torch.load(weights_path))
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Stop at the last convolution layer
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        return x

def extract_patches(img_tensor):
    # Assuming the image tensor is (1, C, 256, 16)
    patches = img_tensor.unfold(2, 16, 16)  # Unfold along the height
    patches = patches.contiguous().view(-1, 3, 16, 16)  # Flatten into patches
    return patches

# # v1,计算16个patch
# def cosine_similarity_matrix(features):
#     features = features.view(features.size(0), -1)  # Flatten features for similarity calculation
#     norm_features = torch.nn.functional.normalize(features, dim=1)
#     return torch.mm(norm_features, norm_features.t())

# def process_dataset(image_dir, weights_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 16)),  # Resize to be able to split into 16 patches of 16x16
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     feature_extractor = ResNetFeatures(weights_path).cuda()

#     all_similarity_matrices = []
#     for img_path in image_paths:  
#         image = Image.open(img_path).convert("RGB")
#         image = transform(image).unsqueeze(0).cuda()  # Add batch dimension
#         patches = extract_patches(image)
#         features = feature_extractor(patches)
#         similarity_matrix = cosine_similarity_matrix(features)
#         all_similarity_matrices.append(similarity_matrix)
#         torch.set_printoptions(threshold=5000, linewidth=200)
#         # print("all_similarity_matrices:",all_similarity_matrices)

#     # Average the similarity matrices across all images
#     average_similarity_matrix = torch.mean(torch.stack(all_similarity_matrices), dim=0)
#     compressed_matrix = torch.mean(average_similarity_matrix, dim=0)  # 计算每一列的平均值
#     compressed_matrix = compressed_matrix.unsqueeze(0)  # 增加一个维度，使其形状变为[1, 16]
#     return compressed_matrix


#v2，计算之后k个patch
def partial_cosine_similarity_matrix(features, k):
    # 假设features的形状为 [num_patches, channels, height, width]
    num_patches = features.shape[0]
    features = features.view(num_patches, -1)  # 将features展平成2D张量
    normalized_features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.zeros((num_patches, num_patches), device=features.device)
    for i in range(num_patches):
        end_index = min(i + k + 1, num_patches)
        similarity_matrix[i, i:end_index] = torch.mm(normalized_features[i:i+1], normalized_features[i:end_index].t())
    return similarity_matrix

#加入flops计算
# def partial_cosine_similarity_matrix(features, k):
#     # 假设features的形状为 [num_patches, channels, height, width]
#     num_patches = features.shape[0]
    
#     # 展平为 2D 张量
#     features = features.view(num_patches, -1)  # 将features展平成2D张量
#     feature_dim = features.shape[1]  # 特征的维度
    
#     # 正则化
#     normalized_features = F.normalize(features, p=2, dim=1)
    
#     # 初始化相似度矩阵
#     similarity_matrix = torch.zeros((num_patches, num_patches), device=features.device)
    
#     # 计算 FLOPs
#     total_flops = 0
#     for i in range(num_patches):
#         end_index = min(i + k + 1, num_patches)
        
#         # 每次矩阵乘法的计算量：num_patches * feature_dim * (end_index - i)
#         flops_for_this_step = feature_dim * (end_index - i) * feature_dim
#         total_flops += flops_for_this_step
        
#         # 计算相似度
#         similarity_matrix[i, i:end_index] = torch.mm(normalized_features[i:i+1], normalized_features[i:end_index].t())

#     # 输出总 FLOPs
#     normalization_flops = num_patches * feature_dim * 2  # 每个元素的正则化有 2 * feature_dim 次运算
#     total_flops += normalization_flops  # 将正则化的 FLOPs 加入总 FLOPs
#     print(f"Total FLOPs for partial_cosine_similarity_matrix: {total_flops} FLOPs")

#     return similarity_matrix


def process_dataset(image_dir, weights_path, k=16, d_state=16):
    transform = transforms.Compose([
        transforms.Resize((256, 16)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    feature_extractor = ResNetFeatures(weights_path).cuda()

    all_similarity_matrices = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).cuda()
        patches = extract_patches(image)
        features = feature_extractor(patches)
        
        similarity_matrix = partial_cosine_similarity_matrix(features, k)
        all_similarity_matrices.append(similarity_matrix)
    # print("all_similarity_matrices:", all_similarity_matrices)

    average_similarity_matrix = torch.mean(torch.stack(all_similarity_matrices), dim=0)
    
    non_zero_counts = (average_similarity_matrix != 0).sum(dim=0).float()
    non_zero_sums = average_similarity_matrix.sum(dim=0)
    
    # 避免除以零
    non_zero_counts[non_zero_counts == 0] = 1
    
    compressed_matrix = non_zero_sums / non_zero_counts

    if compressed_matrix.size(0) != d_state:
        raise ValueError("compressed_matrix size does not match d_state")

    compressed_matrix = compressed_matrix.unsqueeze(0)
    # print("compressed_matrix:", compressed_matrix)
    return compressed_matrix

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, d_state):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim,  d_state)

    def forward(self, x):
        N, seq_len, _ = x.shape

        # 分头处理
        values = self.values(x).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(x).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(x).view(N, seq_len, self.heads, self.head_dim)

        # 计算注意力得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # 应用注意力得分到值上
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
    
#v1版本，compressed_matrix只提供形状，不参与计算
class DynamicPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(DynamicPositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        self._init_pos_embedding()

    def _init_pos_embedding(self):
        position = torch.arange(0, self.max_len).unsqueeze(1)
        # print("position.shape:", position.shape)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        # print("div_term.shape:", div_term.shape)
        self.pos_embed[:, :, 0::2] = torch.sin(position * div_term)
        # print("self.pos_embed1.shape:", self.pos_embed.shape)
        self.pos_embed[:, :, 1::2] = torch.cos(position * div_term)
        # print("self.pos_embed2.shape:", self.pos_embed.shape)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        # print("v1版本，compressed_matrix只提供形状，不参与计算")
        # Fetch all position encodings
        full_pos_encoding = self.pos_embed.squeeze(0)  # Shape: [max_len, d_model]
        # Reduce each position's d_model-dimensional vector to a single value
        compressed_pos_encoding = full_pos_encoding.mean(dim=1, keepdim=True)  # Shape: [max_len, 1]
        # print("compressed_pos_encoding.shape:", compressed_pos_encoding.shape)
        # Reshape to [1, max_len] to match expected output
        return compressed_pos_encoding.transpose(0, 1)  # Shape: [1, max_len]

# v2 让compressed_matrix参与计算（+和*）
# class DynamicPositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=16):
#         super(DynamicPositionEncoding, self).__init__()
#         # print("d_model:",d_model)
#         self.d_model = d_model
#         self.max_len = max_len
#         self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
#         self._init_pos_embedding()

#     def _init_pos_embedding(self):
#         position = torch.arange(0, self.max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
#         # print("div_term.shape:", div_term.shape)
#         # print("div_term:", div_term)
#         self.pos_embed[:, :, 0::2] = torch.sin(position * div_term)
#         # print("self.pos_embed1.shape:", self.pos_embed.shape)
#         self.pos_embed[:, :, 1::2] = torch.cos(position * div_term)
#         # print("self.pos_embed2.shape:", self.pos_embed.shape)

#     def forward(self, x, compressed_matrix):
#         full_pos_encoding = self.pos_embed.squeeze(0)  # Shape: [max_len, d_model]
#         modified_pos_encoding = full_pos_encoding * compressed_matrix
#         compressed_pos_encoding = modified_pos_encoding.mean(dim=1, keepdim=True)  # Shape: [max_len, 1]
#         # Reshape to [1, max_len] to match expected output
#         return compressed_pos_encoding.transpose(0, 1)  # Shape: [1,    max_len]


############################################################################################################################################################################


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x
  
    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2,
            dt_rank="auto",
            # ======================
            dropout=0.,
            conv_bias=True,
            bias=False,
            dtype=None,
            # ======================
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model) #d_inner代表
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt')  # (K=4, D, N)
        # print("self.A_logs:", self.A_logs)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj
    
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     # S4D real initialization
    #     A = repeat(
    #         torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
    #         "n -> d n",
    #         d=d_inner,
    #     ).contiguous()
    #     A_log = torch.log(A)  # Keep A_log in fp32
    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies)
    #         if merge:
    #             A_log = A_log.flatten(0, 1)
    #     A_log = nn.Parameter(A_log)
    #     A_log._no_weight_decay = True
    #     return A_log

    #版本1：只加入相似度
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset('data/cutting/images', weights_path=weights_path).view(1, -1)  # 确保是1行N列的形式

    #     # 检查并调整compressed_matrix的尺寸
    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

    #     # 重复compressed_matrix以适应d_inner维度
    #     A = repeat(
    #         compressed_matrix,
    #         "1 n -> d n",
    #         d=d_inner
    #     ).contiguous()
        
    #     A_log = torch.log(A).to(device)  # 转换到适当的设备并取对数
    #     # print("A_log:", A_log.shape)

    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies)
    #         if merge:
    #             A_log = A_log.flatten(0, 1)

    #     A_log = nn.Parameter(A_log, requires_grad=False)
    #     A_log._no_weight_decay = True
    #     return A_log

    # 版本2：位置向量与相似度矩阵相乘后进行对数变换
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认使用GPU，如果可用的话

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)  # 确保是1行N列的形式并且移到正确的设备

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

    #     positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
    #     mul_matrix = compressed_matrix * positional_vector.unsqueeze(0)  # 乘以位置向量

    #     A = repeat(mul_matrix, "1 n -> d n", d=d_inner).contiguous()

    #     A_log = torch.log(A + 1).to(device)  # 应用对数变换并确保在同一设备

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_log))

    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies)
    #         if merge:
    #             A_log = A_log.flatten(0, 1)

    #     A_log = nn.Parameter(A_log)
    #     A_log._no_weight_decay = True
    #     return A_log
    
    # 版本3：位置向量与相似度矩阵相加后进行对数变换
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认使用GPU，如果可用的话

        image_dir = 'data/cutting/images'
        weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
        compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)  # 确保是1行N列的形式并且移到正确的设备

        if compressed_matrix.size(1) != d_state:
            raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

        positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        mul_matrix = compressed_matrix + positional_vector.unsqueeze(0)  # 乘以位置向量

        A = repeat(mul_matrix, "1 n -> d n", d=d_inner).contiguous()

        A_log = torch.log(A + 1).to(device)  # 应用对数变换并确保在同一设备

        with open(file_path, 'w') as file:
            file.write(str(A_log))

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    #补充实验 位置向量与相似度矩阵相加后不进行对数变换
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认使用GPU，如果可用的话

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)  # 确保是1行N列的形式并且移到正确的设备

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

    #     positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
    #     mul_matrix = compressed_matrix + positional_vector.unsqueeze(0)  # 乘以位置向量

    #     A_log = repeat(mul_matrix, "1 n -> d n", d=d_inner).contiguous()

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_log))

    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies)
    #         if merge:
    #             A_log = A_log.flatten(0, 1)

    #     A_log = nn.Parameter(A_log)
    #     A_log._no_weight_decay = True
    #     return A_log
    
    # 版本4：位置向量与相似度矩阵相乘后不进行对数变换
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认使用GPU，如果可用的话

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)  # 确保是1行N列的形式并且移到正确的设备

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

    #     positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
    #     mul_matrix = compressed_matrix * positional_vector.unsqueeze(0)  # 乘以位置向量

    #     A_log = repeat(mul_matrix, "1 n -> d n", d=d_inner).contiguous()

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_log))

    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies) 
    #         if merge:
    #             A_log = A_log.flatten(0, 1)

    #     A_log = nn.Parameter(A_log)
    #     A_log._no_weight_decay = True
    #     return A_log

    # 版本5：拼接后通过全连接层
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认使用GPU，如果可用的话

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)  # 确保是1行N列的形式并且移到正确的设备

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError(f"compressed_matrix size {compressed_matrix.size(1)} does not match d_state {d_state}")

    #     positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
    #     concatenated = torch.cat([compressed_matrix, positional_vector.unsqueeze(0)], dim=1)  # 拼接操作
    #     # print("concatenated:", concatenated)

    #     # 定义并通过全连接层
    #     mlp = nn.Sequential(
    #         nn.Linear(2 * d_state, d_state),  # 第一个全连接层
    #         nn.ReLU(),  # 非线性激活层
    #         nn.Linear(d_state, d_state)  # 第二个全连接层，输出维度与输入相同
    #     ).to(device)

    #     A_mlp = mlp(concatenated)  # 通过MLP处理拼接后的矩阵
    #     print("A_mlp:", A_mlp)
    #     A_matrix = repeat(A_mlp, "1 n -> d n", d=d_inner).contiguous()

    #     # A_log = torch.log(A_matrix + 1).to(device)  # 对数变换增加稳定性，加1保证非负

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_matrix))

    #     if copies > 1:
    #         A_matrix = repeat(A_matrix, "d n -> r d n", r=copies)
    #         if merge:
    #             A_matrix = A_matrix.flatten(0, 1)

    #     A_matrix = nn.Parameter(A_matrix)
    #     A_matrix._no_weight_decay = True
    #     return A_matrix

    # 版本6：交互式注意力机制,拼接后通过Self-Attention
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError("compressed_matrix size does not match d_state")

    #     positional_vector = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
    #     concatenated = torch.cat([compressed_matrix, positional_vector.unsqueeze(0)], dim=1)
    #     concatenated = concatenated.unsqueeze(1)  # 调整为 [1, 1, 2 * d_state]

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     attention_layer = SelfAttention(embed_size=2 * d_state, heads=8, d_state=d_state).to(device)
    #     concatenated = concatenated.to(device)
    #     A_attention = attention_layer(concatenated)  # 确保 concatenated 的形状是 [batch_size, seq_len, embed_size]
    #     A_attention = A_attention.squeeze(0).unsqueeze(0)
    #     A_log = repeat(A_attention, "1 1 n -> d n", d=d_inner).contiguous()

    #     if A_log.shape != (d_inner, d_state):
    #         raise ValueError(f"Generated A_log shape {A_log.shape} does not match expected shape ({d_inner}, {d_state})")

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_log))

    #     if copies > 1:
    #         A_log = repeat(A_log, "d n -> r d n", r=copies)
    #         if merge:
    #             A_log = A_log.flatten(0, 1)

    #     A_log = nn.Parameter(A_log)
    #     A_log._no_weight_decay = True
    #     return A_log

    # 版本7：v1版本，贴合v1的动态位置编码
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError("compressed_matrix size does not match d_state")

    #     # 初始化动态位置编码层
    #     pos_encoder = DynamicPositionEncoding(d_state).to(device)
    #     positional_encoding = pos_encoder(compressed_matrix)
    #     # print("Shape of positional_encoding:", positional_encoding.shape)
    #     # 根据d_inner进行重复扩展
    #     A_matrix = repeat(positional_encoding, "1 n -> d n", d=d_inner).contiguous()
    #     # print("Shape of A:", A_matrix.shape)
    #     if copies > 1:
    #         # 根据copies进行重复扩展
    #         A_matrix = repeat(A_matrix, "d n -> r d n", r=copies)
    #         if merge:
    #             # 如果需要合并，将A平铺成一维
    #             A_matrix = A_matrix.flatten(0, 1)
    #     A_matrix = nn.Parameter(A_matrix)
    #     A_matrix._no_weight_decay = True
    #     return A_matrix

    # 版本7：v2版本，贴合v2的动态位置编码
    # @staticmethod
    # def A_log_init(d_state, d_inner, copies=1, device=None, merge=True, file_path='model/short_range_vmamba_unet/A_init_log.txt'):
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     image_dir = 'data/cutting/images'
    #     weights_path = '/home/czh/vmamba-unet/model/short_range_vmamba_unet/resnet50.pth'
    #     compressed_matrix = process_dataset(image_dir, weights_path).view(1, -1).to(device)

    #     if compressed_matrix.size(1) != d_state:
    #         raise ValueError("compressed_matrix size does not match d_state")

    #     # 初始化动态位置编码层 
    #     pos_encoder = DynamicPositionEncoding(d_state).to(device)
    #     positional_encoding = pos_encoder(None, compressed_matrix)
    #     # 根据d_inner进行重复扩展
    #     A_matrix = repeat(positional_encoding, "1 n -> d n", d=d_inner).contiguous()
    #     if copies > 1:
    #         # 根据copies进行重复扩展
    #         A_matrix = repeat(A_matrix, "d n -> r d n", r=copies)
    #         if merge:
    #             # 如果需要合并，将A平铺成一维
    #             A_matrix = A_matrix.flatten(0, 1)

    #     with open(file_path, 'w') as file:
    #         file.write(str(A_matrix))
    #     A_matrix = nn.Parameter(A_matrix)
    #     A_matrix._no_weight_decay = True
    #     return A_matrix

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor):
        # print("here is selective start")
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L)  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    forward_core = forward_corev1 #forward_corev1是用来计算的，forward_corev0是用来debug的

    # forward_core = forward_corev0

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            use_checkpoint: bool = False,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# 这段代码定义了一个名为 VSSM 的类，该类继承自 PyTorch 的 nn.Module。VSSM 类是一个深度学习模型，具有以下主要特性：

# __init__ 方法：初始化模型的参数和层。这包括：

# num_classes：分类任务的类别数量。
# depths 和 dims：定义模型的层数和每层的维度。
# d_state、dt_rank、ssm_ratio、attn_drop_rate、drop_rate、drop_path_rate 和 mlp_ratio：这些参数用于配置模型的不同部分。
# norm_layer：用于标准化的层。
# if_down：一个布尔值，表示是否进行下采样。
# use_checkpoint：一个布尔值，表示是否使用检查点（checkpoint）。
# _init_weights 方法：用于初始化模型的权重。

# forward 方法：定义模型的前向传播过程。这包括通过所有层的循环，并在需要时进行下采样。

# 注意，这个类的具体功能和行为取决于 VMambaLayer 类的实现，以及 _make_downsample 函数的实现。
class VSSM(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            depths=[2, 2, 9, 2], #这四个分别代表了每个stage的block数量，分成四个stage是为了方便下采样
            dims=[96, 192, 384, 768],
            # =========================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0, # SSM的扩展比例
            attn_drop_rate=0.,
            # =========================
            drop_rate=0.,
            drop_path_rate=0.1,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            if_down=True, # if_downsample
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.if_down = if_down
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int): #如果dims是一个整数，那么就将dims扩展成一个列表
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        print("dims:", dims)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(VMambaLayer(
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                if_down=if_down,
                if_sample=not self.num_layers == 1,
                if_linear_embed=if_down and i_layer == 0
            ))

        if if_down:
            self.last_downsample = _make_downsample(
                self.dims[-1],
                self.dims[-1] * 2,
                norm_layer=norm_layer,
            )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, down_feats=None):
        if self.if_down:
            all_feat = []

        for layer in self.layers:
            x = layer(x, down_feats.pop() if down_feats is not None else None)
            if self.if_down:
                all_feat.append(x)
        if self.if_down:
            x = self.last_downsample(x)
        # x = self.classifier(x)
        return (x, all_feat) if self.if_down else x


def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        Permute(0, 3, 1, 2),
        nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
        Permute(0, 2, 3, 1),
        norm_layer(out_dim),
    )


def _make_upsample(dim=192, out_dim=96, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        Permute(0, 3, 1, 2),
        nn.ConvTranspose2d(dim, out_dim, kernel_size=2, stride=2),
        Permute(0, 2, 3, 1),
        norm_layer(out_dim),
    )


class VMambaLayer(nn.Module):

    def __init__(self,
                 dim=96,
                 depth=2,
                 drop_path=[0.1, 0.1],
                 use_checkpoint=False,
                 norm_layer=nn.LayerNorm,
                 # ===========================
                 d_state=16,
                 dt_rank="auto",
                 ssm_ratio=2.0,
                 attn_drop_rate=0.0,
                 # ===========================
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 if_down=True,
                 if_sample=True,
                 if_linear_embed=False,
                 **kwargs, ) -> None:
        super().__init__()

        assert depth == len(drop_path) and dim % 2 == 0
        blocks = []
        if if_linear_embed:
            sample = nn.Linear(int(dim / 2), dim)
        else:

            if if_sample:
                self.linear = nn.Linear(dim * 2, dim) if not if_down else nn.Identity()
                sample = _make_downsample(int(dim / 2), dim, norm_layer=norm_layer) if if_down \
                    else _make_upsample(dim * 2, dim, norm_layer=norm_layer)
            else:
                sample = nn.Identity()
        self.sample = sample
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                **kwargs,
            ))
        self.blocks = nn.Sequential(*blocks, )

    def forward(self, x, down_feat=None):
        x = self.sample(x)
        if down_feat is not None:
            x = torch.cat((x, down_feat), dim=-1)
            x = self.linear(x)

        return self.blocks(x)





