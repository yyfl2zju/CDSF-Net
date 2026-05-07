from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class LinearSpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class LinearChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x.mean(dim=(2, 3))
        gate = self.fc(desc).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class WindowSelfAttention2D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape

        xw = x.permute(0, 2, 3, 1)
        xw = xw.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5)
        xw = xw.reshape(-1, ws * ws, c)

        x_norm = self.norm(xw)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        xw = xw + self.drop(out)

        xw = xw.view(b, hp // ws, wp // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5)
        xw = xw.reshape(b, hp, wp, c).permute(0, 3, 1, 2)

        if pad_h > 0 or pad_w > 0:
            xw = xw[:, :, :h, :w]
        return xw


class _PointwiseContext(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(1, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CSFRBlock(nn.Module):
    def __init__(self, low_ch: int, high_ch: int) -> None:
        super().__init__()
        self.high_lsa = LinearSpatialAttention(high_ch)
        self.low_lca = LinearChannelAttention(low_ch)

        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_ch),
        )

        self.ds_main = DepthwiseSeparableConv(high_ch, high_ch, stride=2)
        self.ds_res = nn.Sequential(
            DepthwiseSeparableConv(high_ch, high_ch, stride=1),
            DepthwiseSeparableConv(high_ch, high_ch, stride=2),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.GELU(),
        )

    def _shape_align(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = target_hw
        if x.shape[-2:] == (th, tw):
            return x

        while x.shape[-2] >= th * 2 and x.shape[-1] >= tw * 2:
            x = F.relu(self.ds_main(x) + self.ds_res(x), inplace=True)
            if x.shape[-2:] == (th, tw):
                return x

        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        high_ref = high_feat + self.high_lsa(high_feat)
        low_ref = self.low_lca(low_feat)
        low_ref = self.low_to_high(low_ref)
        low_ref = self._shape_align(low_ref, high_ref.shape[-2:])
        fused = high_ref + low_ref
        return self.out_proj(fused)


class DASFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 4,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.local_window_attn = WindowSelfAttention2D(
            channels,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.global_proj_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_proj_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.local_ctx = _PointwiseContext(channels, reduction=reduction)
        self.global_ctx = _PointwiseContext(channels, reduction=reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        global_map: torch.Tensor,
        local_map: torch.Tensor,
        return_weight: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if global_map.shape[-2:] != local_map.shape[-2:]:
            global_map = F.interpolate(
                global_map,
                size=local_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        f_local = self.local_window_attn(local_map)
        f_global = self.global_proj_1x1(global_map) + self.global_proj_3x3(global_map)

        f_in = f_local + f_global
        l_ctx = self.local_ctx(f_in)
        g_ctx = self.global_ctx(F.adaptive_avg_pool2d(f_in, 1))
        weight = self.sigmoid(l_ctx + g_ctx)

        out = weight * f_global + (1.0 - weight) * f_local
        if return_weight:
            return out, weight
        return out, None


class CDSFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_dim: int = 256,
        fuse_dim: int = 512,
        patch_topk: int = 16,
        dasf_heads: int = 4,
        dasf_window_size: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_topk = patch_topk

        backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.stages = backbone.features

        self.csfr_12 = CSFRBlock(low_ch=256, high_ch=512)
        self.csfr_23 = CSFRBlock(low_ch=512, high_ch=1024)

        self.local_proj_map = nn.Conv2d(512, patch_dim, kernel_size=1)
        self.global_proj_map = nn.Conv2d(1024, patch_dim, kernel_size=1)
        self.dasf = DASFBlock(
            channels=patch_dim,
            num_heads=dasf_heads,
            window_size=dasf_window_size,
            reduction=4,
        )

        self.patch_score = nn.Linear(patch_dim, 1)
        self.patch_norm = nn.LayerNorm(patch_dim)

        self.global_token_proj = nn.Sequential(
            nn.Linear(patch_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patch_token_proj = nn.Sequential(
            nn.Linear(patch_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(fuse_dim * 2, fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, 2),
            nn.Softmax(dim=1),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(fuse_dim * 2, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _forward_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        stage_outs = []
        for idx, block in enumerate(self.stages):
            x = block(x)
            if idx % 2 == 1:
                stage_outs.append(x)
        return tuple(stage_outs)

    def _patch_pool(self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        channels = tokens.shape[2]
        k = min(max(1, int(self.patch_topk)), scores.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)
        topk_attn = F.softmax(topk_vals, dim=1)

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, channels)
        topk_tokens = torch.gather(tokens, dim=1, index=gather_idx)
        pooled = torch.sum(topk_tokens * topk_attn.unsqueeze(-1), dim=1)
        return self.patch_norm(pooled)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
    ):
        _, s1, s2, s3 = self._forward_stages(x)

        s2_ref = self.csfr_12(s1, s2)
        s3_ref = self.csfr_23(s2_ref, s3)

        local_map = self.local_proj_map(s2_ref)
        global_map = self.global_proj_map(s3_ref)
        fused_map, dasf_weight = self.dasf(global_map, local_map, return_weight=True)

        tokens = fused_map.flatten(2).transpose(1, 2)
        token_scores = self.patch_score(tokens).squeeze(-1)
        patch_vec = self._patch_pool(tokens, token_scores)

        tok_global = self.global_token_proj(F.adaptive_avg_pool2d(global_map, 1).flatten(1))
        tok_patch = self.patch_token_proj(patch_vec)

        gate = self.fuse_gate(torch.cat([tok_global, tok_patch], dim=1))
        fused_feat = self.embed_proj(
            torch.cat(
                [
                    gate[:, 0:1] * tok_global,
                    gate[:, 1:2] * tok_patch,
                ],
                dim=1,
            )
        )
        logits = self.classifier(fused_feat)
        aux_dict = {
            "dasf_weight_mean": dasf_weight.mean().detach(),
            "fusion_gate_mean": gate.mean(dim=0).detach(),
        }

        if return_aux and return_features:
            return logits, fused_feat, aux_dict
        if return_aux:
            return logits, aux_dict
        if return_features:
            return logits, fused_feat
        return logits
