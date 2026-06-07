import math
from functools import partial
import torch
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F
from ssim_improments import ImprovedSeisEncoder, ConvSeisAligner


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, seis_cond):
        x = self.norm(x)
        return self.fn(x, seis_cond)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),  # 深度可分离卷积
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class EnhancedTimeMLP(nn.Module):

    def __init__(self, dim=32, time_dim=None):
        super().__init__()
        if time_dim is None:
            time_dim = dim * 4
        learned_sinusoidal_dim = 32
        self.learned_fourier = RandomOrLearnedSinusoidalPosEmb(
            learned_sinusoidal_dim, is_random=False
        )
        fourier_dim = learned_sinusoidal_dim + 1  # 33
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim + 2, time_dim),  # +2: [t², 1/(1-t)]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        fourier = self.learned_fourier(t)           # (B, 33)
        t_sq = (t ** 2).unsqueeze(-1)               # (B, 1)
        t_inv = (1.0 / (1.0 - t + 1e-6)).unsqueeze(-1)  # (B, 1)  高t区爆炸增长
        extra = torch.cat([t_sq, t_inv], dim=-1)    # (B, 2)
        return self.mlp(torch.cat([fourier, extra], dim=-1))


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        cond_dim = 0
        if exists(time_emb_dim):
            cond_dim += int(time_emb_dim)
        if exists(classes_emb_dim):
            cond_dim += int(classes_emb_dim)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_out * 2)
        ) if cond_dim > 0 else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.GroupNorm(groups, dim_out)
        ) if dim != dim_out else nn.Identity()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim_out // 4, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        se_weight = self.se(h)
        h = h * se_weight

        return h + self.res_conv(x)


class UnetConcat(nn.Module):

    def __init__(
            self,
            dim=64,
            num_classes=None,
            cond_drop_prob=0.5,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            seis_channels=64,
            resnet_block_groups=8,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        self.channels = channels
        self.seis_channels = seis_channels

        init_dim = default(init_dim, dim)

        self.seis_aligner = ConvSeisAligner(seis_channels, init_dim)
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, init_dim, kernel_size=7, stride=1, padding=3),
            LayerNorm(init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 70->68
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 68->66
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=0),  # 66->64
        )

        self.seis_layer_conv = nn.Sequential(
            nn.Conv2d(seis_channels, seis_channels, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 70->68
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 68->66
            nn.Conv2d(seis_channels, seis_channels, 3, padding=0),  # 66->64
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings（增强版: LearnedFourier + 1/(1-t) + t²）
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = True  # 强制使用可学习频率
        self.time_mlp = EnhancedTimeMLP(dim=dim, time_dim=time_dim)

        # class embeddings (可选)
        self.classes_emb = nn.Embedding(num_classes, dim) if num_classes is not None and num_classes > 0 else None
        self.null_classes_emb = nn.Parameter(torch.randn(dim)) if num_classes is not None and num_classes > 0 else None

        classes_dim = dim * 4 if (num_classes is not None and num_classes > 0) else None
        self.classes_dim = classes_dim
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        ) if classes_dim is not None else None

        # 🚀 地震特征提取器：使用ImprovedSeisEncoder
        self.feature_net = ImprovedSeisEncoder(
            in_channels=5,
            out_channels=seis_channels,
            conv1d_kernel_size=25,
            time_pool_learnable=True,
            use_spatial_attention=True,
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ImprovedFeatureFusion(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_mlp2 = FeedForward(mid_dim, mult=4)  # 再加一层 MLP

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ImprovedFeatureFusion(dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        # 改进的最终输出层
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_mlp = FeedForward(dim, mult=2)  # 添加特征增强
        self.final_shape_conv = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1,
                                                   padding=0, output_padding=0)
        self.final_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, t, seis=None):
        seis_features = self.feature_net(seis)

        # (B, 64, 70, 70) -> (B, 64, 64, 64)
        seis_layer_features = self.seis_layer_conv(seis_features)
        x = self.init_conv(x)  # (B, 1, 70, 70) -> (B, 64, 64, 64)
        r = x.clone()
        t = self.time_mlp(t)
        h = []

        # Encoder
        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t, t)
            h.append(x)
            aligned_seis = self.seis_aligner(seis_layer_features, mode='encoder', level_idx=idx)
            x = block2(x, t, t)
            x = attn(x, aligned_seis)
            h.append(x)
            x = downsample(x)

        # Bottleneck - 增强全局特征提取
        x = self.mid_block1(x, t, t)
        x = self.mid_block2(x, t, t)
        x = x + self.mid_mlp2(x)  # 再次增强

        # Decoder
        for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, t)
            aligned_seis = self.seis_aligner(seis_layer_features, mode='decoder', level_idx=idx)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, t)
            x = attn(x, aligned_seis)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, t)
        x = x + self.final_mlp(x)  # 特征增强
        x = self.final_shape_conv(x)
        x = x + self.final_refine(x)  # 最终细化
        return self.final_conv(x)

    def forward_with_cfg(self, x, t, y, cfg_scale, is_train_student=False, seis=None):
        if is_train_student:
            t = t.repeat(2)
        else:
            t = t.repeat(x.shape[0])

        # 有条件预测（地震特征正常）
        logits = self.forward(x, t, seis=seis)

        if cfg_scale == 1:
            return logits

        # 无条件预测（gate_mask=0 强制门控关闭，等效于无条件生成）
        null_logits = self.forward(x, t, seis=seis)

        # CFG 公式：output = null + cfg_scale * (cond - null)
        return null_logits + (logits - null_logits) * cfg_scale


class GatedSeisFusion(nn.Module):
    """
    空间门控特征融合模块（替换原 ImprovedFeatureFusion）

    每个空间位置独立学习"该用多少地震信息":
      - 反射界面/断层区 → gate→1（多参考地震特征）
      - 均匀介质区     → gate→0（信任UNet速度先验）

    公式: output = proj(cat(seis,x)) * gate + x * (1 - gate)
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 5, padding=2),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            LayerNorm(dim)
        )

    def forward(self, x, seis, gate_mask=None):
        combined = torch.cat((seis, x), dim=1)
        gate = self.gate(combined)
        fused = self.proj(combined)
        return fused * gate + x * (1 - gate)

ImprovedFeatureFusion = GatedSeisFusion


if __name__ == "__main__":
    x = torch.randn(1, 1, 70, 70)
    seis = torch.randn(1, 5, 1000, 70)
    ts = torch.randint(low=1, high=1000, size=(x.shape[0],))
    unet = UnetConcat(
        channels=1,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=1,
        seis_channels=64,
    )

    out = unet.forward(x=x, t=ts, seis=seis)