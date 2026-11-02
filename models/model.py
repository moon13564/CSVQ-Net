from __future__ import annotations

import os
import inspect
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import timm
from transformers import BertTokenizer
from timm.models.hub import download_cached_file

from models_blip.med import BertConfig, BertModel
from models_blip.vit import VisionTransformer, interpolate_pos_embed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MED_CONFIG = PROJECT_ROOT / "models" / "med_config.json"
DEFAULT_BLIP_CKPT = PROJECT_ROOT / "models" / "model_base_retrieval_coco.pth"


def _init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def _create_vit(vit: str, image_size: int):
    if vit == "base":
        width, depth, heads, drop = 768, 12, 12, 0.0
    elif vit == "large":
        width, depth, heads, drop = 1024, 24, 16, 0.1
    else:
        raise ValueError("vit must be 'base' or 'large'")

    encoder = VisionTransformer(
        img_size=image_size,
        patch_size=16,
        embed_dim=width,
        depth=depth,
        num_heads=heads,
        use_grad_checkpointing=False,
        ckpt_layer=0,
        drop_path_rate=drop,
    )
    return encoder, width


def _is_url(path: str) -> bool:
    return urlparse(path).scheme in ("http", "https")


def _load_blip_checkpoint(model: nn.Module, path_or_url: str):
    if _is_url(path_or_url):
        filename = download_cached_file(
            path_or_url, check_hash=False, progress=True
        )
        checkpoint = torch.load(filename, map_location="cpu")
    elif os.path.isfile(path_or_url):
        checkpoint = torch.load(path_or_url, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"BLIP checkpoint not found: {path_or_url}\n"
            f"Please put it at: {DEFAULT_BLIP_CKPT}"
        )

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    if "visual_encoder.pos_embed" in state:
        state["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state["visual_encoder.pos_embed"], model.visual_encoder
        )

    current = model.state_dict()
    incompatible = [
        k for k, v in state.items()
        if k in current and current[k].shape != v.shape
    ]
    for k in incompatible:
        del state[k]

    msg = model.load_state_dict(state, strict=False)
    return msg


class BLIPITMBackbone(nn.Module):
    def __init__(
        self,
        med_config_path: str,
        pretrained_path: str,
        image_size: int = 224,
        vit: str = "base",
        embed_dim: int = 256,
    ):
        super().__init__()

        med_config_path = Path(med_config_path)
        if not med_config_path.is_file():
            raise FileNotFoundError(f"med_config.json not found: {med_config_path}")

        self.visual_encoder, vision_width = _create_vit(vit, image_size)
        self.tokenizer = _init_tokenizer()

        cfg = BertConfig.from_json_file(str(med_config_path))
        cfg.encoder_width = vision_width
        self.text_encoder = BertModel(config=cfg, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2)

        msg = _load_blip_checkpoint(self, str(pretrained_path))
        self.checkpoint_missing_keys = list(msg.missing_keys)
        self.checkpoint_unexpected_keys = list(msg.unexpected_keys)

        if self.checkpoint_missing_keys:
            preview = "\n".join(self.checkpoint_missing_keys[:30])
            raise RuntimeError(
                "BLIP checkpoint has missing keys:\n" + preview
            )


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.1, ffn_ratio=4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        query_attention_mask=None,
    ):
        q = self.norm_q(query)
        k = self.norm_kv(key)
        v = self.norm_kv(value)

        y, _ = self.attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = query + y

        if query_attention_mask is not None:
            x = x * query_attention_mask.unsqueeze(-1).to(x.dtype)

        x = x + self.ffn(self.norm_ffn(x))

        if query_attention_mask is not None:
            x = x * query_attention_mask.unsqueeze(-1).to(x.dtype)

        return x


class SemanticBranch(nn.Module):
    def __init__(
        self,
        med_config_path: str = str(DEFAULT_MED_CONFIG),
        blip_pretrained_path: str = str(DEFAULT_BLIP_CKPT),
        image_size: int = 224,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_text_length: int = 64,
        freeze_visual_encoder: bool = True,
        freeze_text_encoder: bool = True,
        train_projection: bool = True,
        unfreeze_vis_layers: int = 0,
        unfreeze_txt_layers: int = 0,
        use_pd_global: bool = True,
        use_abs: bool = True,
        use_mul: bool = True,
        use_global_mlp: bool = True,
        use_po_image_ca: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.max_text_length = max_text_length
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_text_encoder = freeze_text_encoder

        self.use_pd_global = bool(use_pd_global)
        self.use_abs = bool(use_abs)
        self.use_mul = bool(use_mul)
        self.use_global_mlp = bool(use_global_mlp)
        self.use_po_image_ca = bool(use_po_image_ca)

        self.blip = BLIPITMBackbone(
            med_config_path=med_config_path,
            pretrained_path=blip_pretrained_path,
            image_size=image_size,
            vit="base",
            embed_dim=embed_dim,
        )

        self._set_grad(self.blip, False)

        if not freeze_visual_encoder:
            self._set_grad(self.blip.visual_encoder, True)
        if not freeze_text_encoder:
            self._set_grad(self.blip.text_encoder, True)

        self._set_grad(self.blip.vision_proj, train_projection)
        self._set_grad(self.blip.text_proj, train_projection)

        self.unfreeze_vis_layers = int(unfreeze_vis_layers)
        self.unfreeze_txt_layers = int(unfreeze_txt_layers)

        if self.unfreeze_vis_layers or self.unfreeze_txt_layers:
            if (not freeze_visual_encoder) or (not freeze_text_encoder):
                raise ValueError(
                    "unfreeze_vis_layers / unfreeze_txt_layers 不能与 "
                    "freeze_visual_encoder=False / freeze_text_encoder=False 同时使用。"
                )
            self._unfreeze_backbone_tail(
                self.unfreeze_vis_layers, self.unfreeze_txt_layers
            )

        self._vis_use_no_grad = (
            bool(self.freeze_visual_encoder) and self.unfreeze_vis_layers == 0
        )
        self._txt_use_no_grad = (
            bool(self.freeze_text_encoder) and self.unfreeze_txt_layers == 0
        )
        self._partial_unfreeze = bool(
            self.unfreeze_vis_layers or self.unfreeze_txt_layers
        )

        global_terms = 2
        if self.use_pd_global:
            global_terms += 1
        if self.use_abs:
            global_terms += 1
        if self.use_mul:
            global_terms += 1
        self.global_terms = global_terms

        if self.use_global_mlp:
            self.global_semantic_mlp = nn.Sequential(
                nn.Linear(embed_dim * global_terms, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.global_semantic_mlp = None

        if self.use_po_image_ca:
            self.po_image_attention = CrossAttentionBlock(
                embed_dim, num_heads, dropout
            )
        else:
            self.po_image_attention = None

        sem_terms = int(self.use_global_mlp) + int(self.use_po_image_ca)
        if sem_terms < 1:
            raise ValueError(
                "SemanticBranch needs at least one of "
                "use_global_mlp / use_po_image_ca to be True."
            )
        self.sem_terms = sem_terms

        self.semantic_fusion = nn.Sequential(
            nn.Linear(embed_dim * sem_terms, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @staticmethod
    def _set_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _unfreeze_backbone_tail(self, n_vis: int, n_txt: int):
        if n_vis:
            blocks = getattr(self.blip.visual_encoder, "blocks", None)
            if blocks is None:
                raise AttributeError(
                    "visual_encoder 没有 .blocks，无法局部解冻；请检查 vit 实现。"
                )
            if n_vis > len(blocks):
                raise ValueError(
                    f"unfreeze_vis_layers={n_vis} 超过 ViT 深度 {len(blocks)}"
                )
            for blk in list(blocks)[-n_vis:]:
                self._set_grad(blk, True)
            norm = getattr(self.blip.visual_encoder, "norm", None)
            if norm is not None:
                self._set_grad(norm, True)

        if n_txt:
            enc = getattr(self.blip.text_encoder, "encoder", None)
            layers = getattr(enc, "layer", None) if enc is not None else None
            if layers is None:
                raise AttributeError(
                    "text_encoder.encoder.layer 不存在，无法局部解冻 BERT。"
                )
            if n_txt > len(layers):
                raise ValueError(
                    f"unfreeze_txt_layers={n_txt} 超过 BERT 层数 {len(layers)}"
                )
            for lyr in list(layers)[-n_txt:]:
                self._set_grad(lyr, True)

    def backbone_trainable_parameters(self):
        return [
            p for p in self.blip.parameters()
            if p.requires_grad
        ]

    def unfreeze_summary(self) -> Dict[str, int]:
        n_vis = sum(
            p.numel() for p in self.blip.visual_encoder.parameters()
            if p.requires_grad
        )
        n_txt = sum(
            p.numel() for p in self.blip.text_encoder.parameters()
            if p.requires_grad
        )
        return {
            "unfreeze_vis_layers": self.unfreeze_vis_layers,
            "unfreeze_txt_layers": self.unfreeze_txt_layers,
            "trainable_visual_encoder": int(n_vis),
            "trainable_text_encoder": int(n_txt),
        }

    def train(self, mode=True):
        super().train(mode)
        if mode and self._vis_use_no_grad:
            self.blip.visual_encoder.eval()
        if mode and self._txt_use_no_grad:
            self.blip.text_encoder.eval()
        if mode and self._partial_unfreeze:
            self.blip.visual_encoder.eval()
            self.blip.text_encoder.eval()
        self.blip.itm_head.eval()
        return self

    def _tokenize(self, texts: List[str], device):
        return self.blip.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(device)

    @staticmethod
    def _masked_mean(tokens, mask):
        m = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * m).sum(1) / m.sum(1).clamp_min(1.0)

    def encode_image(self, image):
        if self._vis_use_no_grad:
            with torch.no_grad():
                raw = self.blip.visual_encoder(image)
        else:
            raw = self.blip.visual_encoder(image)

        tokens = self.blip.vision_proj(raw)
        g_img = tokens[:, 0, :]
        return tokens, g_img

    def encode_text(self, texts, device):
        enc = self._tokenize(texts, device)

        kwargs = dict(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            return_dict=True,
            mode="text",
        )

        if self._txt_use_no_grad:
            with torch.no_grad():
                out = self.blip.text_encoder(**kwargs)
        else:
            out = self.blip.text_encoder(**kwargs)

        tokens = self.blip.text_proj(out.last_hidden_state)
        g_text = tokens[:, 0, :]
        return tokens, enc.attention_mask, g_text

    def forward(
        self,
        image,
        original_prompt,
        descriptive_prompt,
        return_aux=False,
    ):
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"SemanticBranch expects {self.image_size}x{self.image_size}, "
                f"got {tuple(image.shape[-2:])}"
            )

        img_tokens, g_img = self.encode_image(image)
        po_tokens, po_mask, g_po = self.encode_text(
            original_prompt, image.device
        )

        needs_pd = self.use_pd_global or self.use_abs or self.use_mul
        if needs_pd:
            pd_tokens, pd_mask, g_pd = self.encode_text(
                descriptive_prompt, image.device
            )
        else:
            pd_tokens, pd_mask, g_pd = None, None, None

        f_global = None
        if self.use_global_mlp:
            parts = [g_img, g_po]
            if self.use_pd_global:
                parts.append(g_pd)
            if self.use_abs:
                parts.append((g_po - g_pd).abs())
            if self.use_mul:
                parts.append(g_po * g_pd)
            f_global = self.global_semantic_mlp(torch.cat(parts, dim=-1))

        f_io = None
        if self.use_po_image_ca:
            io_tokens = self.po_image_attention(
                po_tokens,
                img_tokens,
                img_tokens,
                query_attention_mask=po_mask,
            )
            f_io = self._masked_mean(io_tokens, po_mask)

        sem_parts = []
        if f_global is not None:
            sem_parts.append(f_global)
        if f_io is not None:
            sem_parts.append(f_io)

        f_sem = self.semantic_fusion(torch.cat(sem_parts, dim=-1))

        if not return_aux:
            return f_sem

        return f_sem, {
            "F_global": f_global,
            "f_io": f_io,
            "image_tokens_shape": tuple(img_tokens.shape),
            "po_tokens_shape": tuple(po_tokens.shape),
            "pd_tokens_shape": None if pd_tokens is None else tuple(pd_tokens.shape),
            "pd_mask_shape": None if pd_mask is None else tuple(pd_mask.shape),
            "semantic_design": "ablation_switches",
        }


class MultiScaleQualityBlock(nn.Module):
    def __init__(self, channels=128, kernels=(3, 5, 7), dropout=0.1):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    k,
                    padding=k // 2,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            for k in kernels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels * len(kernels),
                channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        multi = torch.cat([b(x) for b in self.branches], dim=1)
        return x + self.fuse(multi)


class _MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class VisualBranch(nn.Module):
    def __init__(
        self,
        backbone="pvt_v2_b2",
        pretrained=True,
        feature_dim=128,
        embed_dim=256,
        kernels=(3, 5, 7),
        dropout=0.2,
        use_ms: bool = True,
        levels: Tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()

        if backbone != "pvt_v2_b2":
            raise ValueError(
                "AGIQA final VisualBranch expects backbone='pvt_v2_b2'."
            )

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        channels = [64, 128, 320, 512]

        self.use_ms = bool(use_ms)
        self.levels = tuple(levels)
        if not self.levels:
            raise ValueError("levels must contain at least one index")
        for i in self.levels:
            if i < 0 or i > 3:
                raise ValueError(f"level index out of range: {i}")

        selected_channels = [channels[i] for i in self.levels]

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feature_dim, 1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.GELU(),
            )
            for c in selected_channels
        ])

        if self.use_ms:
            self.ms = nn.ModuleList([
                MultiScaleQualityBlock(
                    feature_dim, kernels, dropout=0.1
                )
                for _ in self.levels
            ])
        else:
            self.ms = None

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.visual_head = _MLPHead(
            feature_dim * len(self.levels),
            embed_dim,
            dropout,
        )

    @staticmethod
    def _to_nchw_feature(x, feat_size=None):
        if x.ndim == 4:
            if x.shape[1] in (64, 128, 320, 512):
                return x.contiguous()

            if x.shape[-1] in (64, 128, 320, 512):
                return x.permute(0, 3, 1, 2).contiguous()

            raise RuntimeError(
                f"Cannot determine PVT 4D feature layout from shape {tuple(x.shape)}"
            )

        if x.ndim == 3:
            if feat_size is None:
                raise RuntimeError(
                    "PVT returned token features [B,N,C] but feat_size is missing."
                )

            h, w = int(feat_size[0]), int(feat_size[1])
            b, n, c = x.shape

            spatial_tokens = h * w
            if n == spatial_tokens + 1:
                x = x[:, 1:, :]
                n = x.shape[1]

            if n != spatial_tokens:
                raise RuntimeError(
                    f"Cannot reshape PVT tokens: N={n}, feat_size=({h},{w})"
                )

            return (
                x.reshape(b, h, w, c)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        raise RuntimeError(
            f"Unsupported PVT feature rank: shape={tuple(x.shape)}"
        )

    def _forward_pvt_features(self, image):
        patch_out = self.backbone.patch_embed(image)

        feat_size = None
        if isinstance(patch_out, (tuple, list)):
            if len(patch_out) != 2:
                raise RuntimeError(
                    "Unexpected PVT patch_embed return structure: "
                    f"{type(patch_out)} with length {len(patch_out)}"
                )
            x, feat_size = patch_out
        else:
            x = patch_out

            if x.ndim == 4:
                if x.shape[1] in (64, 128, 320, 512):
                    feat_size = (x.shape[2], x.shape[3])
                elif x.shape[-1] in (64, 128, 320, 512):
                    feat_size = (x.shape[1], x.shape[2])

        features = []

        for stage_index, stage in enumerate(self.backbone.stages):
            stage_params = list(inspect.signature(stage.forward).parameters)

            if "feat_size" in stage_params or len(stage_params) >= 2:
                if feat_size is None:
                    raise RuntimeError(
                        f"PVT stage {stage_index} requires feat_size, "
                        "but patch_embed did not provide/infer one."
                    )
                stage_out = stage(x, feat_size)
            else:
                stage_out = stage(x)

            if isinstance(stage_out, (tuple, list)):
                if len(stage_out) != 2:
                    raise RuntimeError(
                        f"Unexpected PVT stage {stage_index} return structure: "
                        f"length={len(stage_out)}"
                    )
                x, feat_size = stage_out
            else:
                x = stage_out

                if x.ndim == 4:
                    if x.shape[1] in (64, 128, 320, 512):
                        feat_size = (x.shape[2], x.shape[3])
                    elif x.shape[-1] in (64, 128, 320, 512):
                        feat_size = (x.shape[1], x.shape[2])

            features.append(
                self._to_nchw_feature(x, feat_size)
            )

        if len(features) != 4:
            raise RuntimeError(
                f"Expected 4 PVT feature levels, got {len(features)}"
            )

        expected_channels = [64, 128, 320, 512]
        actual_channels = [int(f.shape[1]) for f in features]

        if actual_channels != expected_channels:
            raise RuntimeError(
                "Unexpected PVTv2-B2 feature channels: "
                f"{actual_channels}; expected {expected_channels}. "
                f"Shapes={[tuple(f.shape) for f in features]}"
            )

        return features

    def forward(self, image, return_aux=False):
        feats_all = self._forward_pvt_features(image)
        feats = [feats_all[i] for i in self.levels]

        if self.use_ms:
            feats = [
                ms(proj(f))
                for f, proj, ms in zip(feats, self.proj, self.ms)
            ]
        else:
            feats = [
                proj(f)
                for f, proj in zip(feats, self.proj)
            ]

        pooled = [
            self.pool(f).flatten(1)
            for f in feats
        ]

        multi_level = torch.cat(pooled, dim=-1)
        f_vis = self.visual_head(multi_level)

        if not return_aux:
            return f_vis

        return f_vis, {
            "F_vis": f_vis,
            "pooled_multi_level": multi_level,
            "feature_shapes": [tuple(x.shape) for x in feats],
            "visual_design": "ablation_switches",
            "levels": self.levels,
            "use_ms": self.use_ms,
        }


class CrossBranchFusion(nn.Module):
    def __init__(self, dim=256, dropout=0.2, use_gate: bool = True):
        super().__init__()

        self.use_gate = bool(use_gate)

        if self.use_gate:
            self.branch_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, 2),
            )
        else:
            self.branch_gate = None

        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, f_sem, f_vis, return_aux=False):
        if self.use_gate:
            weights = torch.softmax(
                self.branch_gate(
                    torch.cat([f_sem, f_vis], dim=-1)
                ),
                dim=-1,
            )
            w_sem = weights[:, 0:1]
            w_vis = weights[:, 1:2]
        else:
            weights = torch.full(
                (f_sem.size(0), 2),
                0.5,
                device=f_sem.device,
                dtype=f_sem.dtype,
            )
            w_sem = weights[:, 0:1]
            w_vis = weights[:, 1:2]

        mixed = w_sem * f_sem + w_vis * f_vis

        joint = self.proj(
            torch.cat(
                [
                    mixed,
                    f_sem,
                    f_vis,
                    (f_sem - f_vis).abs(),
                ],
                dim=-1,
            )
        )

        if not return_aux:
            return joint

        return joint, {
            "branch_weights": weights,
            "mixed": mixed,
        }


class _Expert(nn.Module):
    def __init__(self, dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, return_aux=False):
        score = self.net(x).squeeze(-1)

        if not return_aux:
            return score

        return score, {
            "expert_scores": score.unsqueeze(-1),
        }


class TopKMoE(nn.Module):
    def __init__(
        self,
        dim=256,
        num_experts=4,
        top_k=2,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError(
                "top_k must be within [1, num_experts]"
            )

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            _Expert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x, return_aux=False):
        logits = self.gate(x)

        top_val, top_idx = torch.topk(
            logits,
            self.top_k,
            dim=-1,
        )

        top_weight = torch.softmax(
            top_val,
            dim=-1,
        )

        all_scores = torch.stack(
            [expert(x) for expert in self.experts],
            dim=-1,
        )

        selected = torch.gather(
            all_scores,
            1,
            top_idx,
        )

        score = (
            selected * top_weight
        ).sum(dim=-1)

        if not return_aux:
            return score

        return score, {
            "gate_logits": logits,
            "topk_index": top_idx,
            "topk_weight": top_weight,
            "expert_scores": all_scores,
        }


class DualBranchAGIQA(nn.Module):
    def __init__(
        self,
        med_config_path: str = str(DEFAULT_MED_CONFIG),
        blip_pretrained_path: str = str(DEFAULT_BLIP_CKPT),
        image_size: int = 224,
        embed_dim: int = 256,
        visual_pretrained: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        use_pd_global: bool = True,
        use_abs: bool = True,
        use_mul: bool = True,
        use_global_mlp: bool = True,
        use_po_image_ca: bool = True,
        unfreeze_vis_layers: int = 0,
        unfreeze_txt_layers: int = 0,
        use_ms: bool = True,
        kernels: Tuple[int, ...] = (3, 5, 7),
        levels: Tuple[int, ...] = (0, 1, 2, 3),
        use_semantic: bool = True,
        use_visual: bool = True,
        fusion_use_gate: bool = True,
        regressor_type: str = "moe",
    ):
        super().__init__()

        self.use_semantic = bool(use_semantic)
        self.use_visual = bool(use_visual)

        if not self.use_semantic and not self.use_visual:
            raise ValueError("At least one of use_semantic/use_visual must be True.")

        if self.use_semantic:
            self.semantic = SemanticBranch(
                med_config_path=med_config_path,
                blip_pretrained_path=blip_pretrained_path,
                image_size=image_size,
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.2,
                max_text_length=64,
                freeze_visual_encoder=True,
                freeze_text_encoder=True,
                train_projection=True,
                use_pd_global=use_pd_global,
                use_abs=use_abs,
                use_mul=use_mul,
                use_global_mlp=use_global_mlp,
                use_po_image_ca=use_po_image_ca,
                unfreeze_vis_layers=unfreeze_vis_layers,
                unfreeze_txt_layers=unfreeze_txt_layers,
            )
        else:
            self.semantic = None

        if self.use_visual:
            self.visual = VisualBranch(
                backbone="pvt_v2_b2",
                pretrained=visual_pretrained,
                feature_dim=128,
                embed_dim=embed_dim,
                kernels=kernels,
                dropout=0.2,
                use_ms=use_ms,
                levels=levels,
            )
        else:
            self.visual = None

        if self.use_semantic and self.use_visual:
            self.fusion = CrossBranchFusion(
                dim=embed_dim,
                dropout=0.2,
                use_gate=fusion_use_gate,
            )
        else:
            self.fusion = None

        regressor_type = str(regressor_type).lower()
        if regressor_type == "moe":
            self.regressor = TopKMoE(
                dim=embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                hidden_dim=128,
                dropout=0.1,
            )
        elif regressor_type == "mlp":
            self.regressor = _Expert(
                dim=embed_dim,
                hidden_dim=128,
                dropout=0.1,
            )
        else:
            raise ValueError(
                f"regressor_type must be 'moe' or 'mlp', got {regressor_type!r}"
            )

    def _hard_branch_weights(self, batch_size, device, dtype, first: bool):
        w = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        w[:, 0 if first else 1] = 1.0
        return w

    def forward(
        self,
        semantic_image,
        visual_image,
        original_prompt,
        descriptive_prompt,
        return_aux=False,
    ):
        if self.use_semantic and self.use_visual:
            if not return_aux:
                f_sem = self.semantic(
                    semantic_image,
                    original_prompt,
                    descriptive_prompt,
                )
                f_vis = self.visual(visual_image)
                f_joint = self.fusion(f_sem, f_vis)
                return self.regressor(f_joint)

            f_sem, sem_aux = self.semantic(
                semantic_image,
                original_prompt,
                descriptive_prompt,
                return_aux=True,
            )
            f_vis, vis_aux = self.visual(
                visual_image,
                return_aux=True,
            )
            f_joint, fusion_aux = self.fusion(
                f_sem,
                f_vis,
                return_aux=True,
            )
            score, moe_aux = self.regressor(
                f_joint,
                return_aux=True,
            )
            return score, {
                "F_sem": f_sem,
                "F_vis": f_vis,
                "F_joint": f_joint,
                "semantic": sem_aux,
                "visual": vis_aux,
                "fusion": fusion_aux,
                "moe": moe_aux,
            }

        if self.use_semantic and not self.use_visual:
            f_sem = self.semantic(
                semantic_image,
                original_prompt,
                descriptive_prompt,
            )
            score = self.regressor(f_sem)

            if not return_aux:
                return score

            bw = self._hard_branch_weights(
                f_sem.size(0), f_sem.device, f_sem.dtype, first=True
            )
            return score, {
                "F_sem": f_sem,
                "F_vis": None,
                "F_joint": f_sem,
                "semantic": None,
                "visual": None,
                "fusion": {"branch_weights": bw},
                "moe": None,
            }

        f_vis = self.visual(visual_image)
        score = self.regressor(f_vis)

        if not return_aux:
            return score

        bw = self._hard_branch_weights(
            f_vis.size(0), f_vis.device, f_vis.dtype, first=False
        )
        return score, {
            "F_sem": None,
            "F_vis": f_vis,
            "F_joint": f_vis,
            "semantic": None,
            "visual": None,
            "fusion": {"branch_weights": bw},
            "moe": None,
        }


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable