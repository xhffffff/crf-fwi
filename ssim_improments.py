"""
ImprovedSeisEncoder - 改进的地震数据编码器

设计原则（基于设计文档）：
1. Conv1d处理时间维度（kernel_size=11），捕捉波形的时间依赖
2. Conv2d处理空间维度（时间×接收器的2D特征图）
3. 震源感知特征聚合（1x1卷积融合，替代简单拼接）
4. 可学习时间池化（替代简单平均池化）
5. 空间注意力机制（自适应选择重要接收器位置）
6. 可选的低频FFT特征（仅前10%频率，默认禁用）

输入: (B, 5, 1000, 70) - 5个震源, 1000时间步, 70接收器
输出: (B, 64, 64, 64) - 64通道, 64x64空间分辨率

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.2, 9.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimePooling(nn.Module):
    def __init__(self, in_channels: int, target_size: int = 70, num_segments: int = 8):
        super().__init__()
        self.target_size = target_size
        self.num_segments = num_segments
        self.in_channels = in_channels

        self.segment_pool = nn.AdaptiveAvgPool1d(num_segments)

        self.segment_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
        )

        self.segment_attn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.conv_pool = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=25, stride=4, padding=12),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
        )

        self.final_pool = nn.AdaptiveAvgPool1d(target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)  time features after time_conv, T ≈ 250
        returns: (B, C, target_size)
        """
        seg = self.segment_pool(x)                     # (B, C, K)
        seg = self.segment_conv(seg)                   # (B, C, K)
        seg_attn = self.segment_attn(seg)              # (B, C, K) soft gate

        x = self.conv_pool(x)                          # (B, C, T/4)
        x = self.final_pool(x)                         # (B, C, target_size)
        x = x * seg_attn.mean(dim=-1, keepdim=True)    # channel reweight from segment attention
        return x


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    自适应选择重要接收器位置，不同接收器位置的重要性不同

    Requirements: 1.3
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()

        # 通道注意力（SE-like）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 2D特征图

        Returns:
            (B, C, H, W) - 注意力加权后的特征图
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        return x

class ImprovedSeisEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 64,
        conv1d_kernel_size: int = 25,
        time_pool_learnable: bool = True,
        use_spatial_attention: bool = True,
    ):
        super().__init__()

        # 验证conv1d_kernel_size在合理范围内 (Requirements 9.2)
        assert 7 <= conv1d_kernel_size <= 35, f"conv1d_kernel_size must be in [7, 35], got {conv1d_kernel_size}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d_kernel_size = conv1d_kernel_size
        self.time_pool_learnable = time_pool_learnable
        self.use_spatial_attention = use_spatial_attention
        self.use_recv_diff = use_recv_diff


        self.time_conv = nn.Sequential(
            # 🔥 第一层就降采样：1000 -> 500
            nn.Conv1d(1, 16, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # 第二层继续降采样：500 -> 250
            nn.Conv1d(16, 32, kernel_size=conv1d_kernel_size, padding=conv1d_kernel_size // 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        self.time_pool = TimePooling(in_channels=32, target_size=70, num_segments=8)

        self.spatial_conv = nn.Sequential(
            # Conv2d 处理 (时间', 接收器) 的2D特征图
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # 接收器维度 70 -> 70 (通过卷积+池化)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )


        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(in_channels=64, reduction=4)
        else:
            self.spatial_attention = None


        self.source_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 64, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


    def forward(self, seis: torch.Tensor) -> torch.Tensor:

        B, S, T, R = seis.shape  # B, 5, 1000, 70

        # 数据预处理：软限幅，保留动态范围
        seis = torch.tanh(seis / 2.0) * 2.0
        x = seis.permute(0, 1, 3, 2).reshape(B * S * R, 1, T)
        x = self.time_conv(x)  # (B*5*70, 32, T')
        x = self.time_pool(x)  # (B*5*70, 32, 70)

        x = x.reshape(B * S, R, 32, 70).permute(0, 2, 3, 1)  # (B*5, 32, 70, 70)
        x = self.spatial_conv(x)  # (B*5, 70, H, W)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        x = x.reshape(B, S * 64, 70, 70)
        x = self.source_fusion(x)  # (B, 64, 70, 70)

        return x




from dataclasses import dataclass
from typing import Optional
import math
from typing import Optional


class LearningRateController:
    def __init__(
        self,
        initial_lr: float = 3e-4,
        warmup_steps: int = 10000,
        cosine_period: int = 80000,
        min_lr_ratio: float = 0.2,
        restart_ratio: float = 0.5,
        decay_ratio: float = 0.5,
        use_warm_restart: bool = False,
        restart_cycle_length: Optional[int] = None,
    ):

        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.cosine_period = cosine_period
        self.min_lr_ratio = min_lr_ratio
        self.restart_ratio = restart_ratio
        self.decay_ratio = decay_ratio
        self.use_warm_restart = use_warm_restart
        self.restart_cycle_length = restart_cycle_length or (cosine_period - warmup_steps)
        
        # 当前有效的初始学习率（可能因重启或衰减而改变）
        self.current_base_lr = initial_lr
        
        # 记录重启和衰减的步数
        self.restart_step: Optional[int] = None
        self.decay_step: Optional[int] = None
        
        # 记录调整历史
        self.adjustment_history = []
    
    def get_lr(self, step: int) -> float:
        # 阶段1: Warmup
        if step < self.warmup_steps:
            return self.current_base_lr * (step / self.warmup_steps)
        
        min_lr = self.current_base_lr * self.min_lr_ratio
        
        # 阶段2: Cosine Annealing (with optional warm-restart)
        if self.use_warm_restart:
            effective_step = (step - self.warmup_steps) % self.restart_cycle_length
            progress = effective_step / self.restart_cycle_length
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (self.current_base_lr - min_lr) * cosine_decay
        elif step < self.cosine_period:
            progress = (step - self.warmup_steps) / (self.cosine_period - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (self.current_base_lr - min_lr) * cosine_decay
        else:
            return min_lr
    
    def restart_lr(self, current_step: int) -> float:
        # 重启学习率至初始值的50%
        self.current_base_lr = self.initial_lr * self.restart_ratio
        self.restart_step = current_step
        
        # 记录调整历史
        self.adjustment_history.append({
            'step': current_step,
            'type': 'restart',
            'new_base_lr': self.current_base_lr,
            'reason': 'Train SSIM停滞，重启学习率突破瓶颈'
        })
        
        # 返回当前学习率
        return self.get_lr(current_step)
    
    def decay_lr(self, current_step: int) -> float:
        # 衰减当前基础学习率至50%
        self.current_base_lr = self.current_base_lr * self.decay_ratio
        self.decay_step = current_step
        
        # 记录调整历史
        self.adjustment_history.append({
            'step': current_step,
            'type': 'decay',
            'new_base_lr': self.current_base_lr,
            'reason': 'Test SSIM下降，降低学习率减缓过拟合'
        })
        
        # 返回当前学习率
        return self.get_lr(current_step)
    
    def get_adjustment_history(self):
        return self.adjustment_history
    
    def reset(self):
        self.current_base_lr = self.initial_lr
        self.restart_step = None
        self.decay_step = None
        self.adjustment_history = []
    
    def __repr__(self):
        return (
            f"LearningRateController("
            f"initial_lr={self.initial_lr}, "
            f"current_base_lr={self.current_base_lr}, "
            f"warmup_steps={self.warmup_steps}, "
            f"cosine_period={self.cosine_period})"
        )


class ConvSeisAligner(nn.Module):
    """专用卷积对齐模块 - 仅使用卷积，不用插值"""

    def __init__(self, base_channels=64, dim=64):
        super().__init__()
        self.base_channels = base_channels
        self.dim = dim

        # ==================== 编码器投影层 ====================
        # 第0层: 64 -> dim, 64x64 -> 64x64
        self.enc_proj_0 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=1, stride=1, padding=0, bias=False),  # 下采样
            nn.GroupNorm(32, dim),
            nn.SiLU()
        )
        # 第1层: 64 -> dim, 64x64 -> 32x32
        self.enc_proj_1 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 下采样
            nn.GroupNorm(32, dim),
            nn.SiLU()
        )

        # 第2层: 64 -> dim*2, 64x64 -> 16x16
        self.enc_proj_2 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.GroupNorm(32, dim * 2),
            nn.SiLU()
        )

        # 第3层: 64 -> dim*4, 64x64 -> 8x8
        self.enc_proj_3 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 16->8
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )

        # ==================== 解码器投影层 ====================
        # 第0解码层: 64 -> dim*8, 64x64 -> 8x8 (与编码器第3层相同)
        self.dec_proj_0 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.Conv2d(dim, dim * 8, kernel_size=3, stride=2, padding=1, bias=False),  # 16->8
            nn.GroupNorm(32, dim * 8),
            nn.SiLU()
        )

        # 第1解码层: 64 -> dim*4, 64x64 -> 16x16 (与编码器第2层相同)
        self.dec_proj_1 = nn.Sequential(
            nn.Conv2d(base_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 32->16
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )

        # 第2解码层: 64 -> dim*2, 64x64 -> 32x32 (与编码器第1层相同)
        self.dec_proj_2 = nn.Sequential(
            nn.Conv2d(base_channels, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.GroupNorm(32, dim * 2),
            nn.SiLU()
        )

        # 第3解码层: 64 -> dim, 64x64 -> 64x64 (初始投影)
        self.dec_proj_3 = self.enc_proj_0


    def forward(self, seis_features, mode='encoder', level_idx=0):
        if mode == 'init':
            # 初始层: dim, 64x64
            return self.init_proj(seis_features)

        elif mode == 'encoder':
            if level_idx == 0:
                # 编码器第1层: dim, 64x64
                return self.enc_proj_0(seis_features)
            elif level_idx == 1:
                # 编码器第2层: dim*1, 32x32
                return self.enc_proj_1(seis_features)
            elif level_idx == 2:
                # 编码器第3层: dim*2, 16x16
                return self.enc_proj_2(seis_features)
            elif level_idx == 3:
                # 编码器第3层: dim*4, 8x8
                return self.enc_proj_3(seis_features)
            else:
                # 默认返回原始特征
                return seis_features

        elif mode == 'decoder':
            if level_idx == 0:
                # 解码器第0层: dim*8, 8x8
                return self.dec_proj_0(seis_features)
            elif level_idx == 1:
                # 解码器第1层: dim*4, 16x16
                return self.dec_proj_1(seis_features)
            elif level_idx == 2:
                # 解码器第2层: dim*2, 32x32
                return self.dec_proj_2(seis_features)
            elif level_idx == 3:
                # 解码器第3层: dim, 64x64
                return self.dec_proj_3(seis_features)
            else:
                # 默认返回原始特征
                return seis_features

        else:
            return seis_features