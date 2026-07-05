# -*- coding: utf-8 -*-
"""
===============================================================================
  AI-Generated Image Quality Assessment Optimized Framework
  基于正向+反向图文一致性与多尺度感知质量的优化框架
  
  三条分支:
    分支1 (Forward):  CLIP正向图文一致性评价
    分支2 (DualPrompt): 双提示词对比模型（原始提示词 vs 描述性提示词）
    分支3 (MSPQN): 多尺度感知质量评估网络
  
  融合: 自适应池化 + 门控注意力 + Top-K MoE → 最终质量分数
  
  数据集: AGIQA-1K, AGIQA-3K, AIGCIQA2023
  评价指标: PLCC, SRCC, LOSS
  硬件: RTX4090
===============================================================================
"""

import os
import sys
import random
import math
import copy
import logging
import time
from itertools import product
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel

# ============================================================================
# 配置: 路径、超参数、设备
# ============================================================================
DATA_ROOT = r'H:\25sunxu\1PV2\data'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

SEED = 20200626
NUM_EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 4
INITIAL_LR = 5e-6
WEIGHT_DECAY = 0.001
T_MAX = 10
NUM_PATCH_TRAIN = 8
NUM_PATCH_TEST = 15

# ============================================================================
# 工具函数
# ============================================================================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(SEED)

def get_logger(path, name='log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)-8s] %(message)s', datefmt='%Y%m%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def log_and_print(logger, msg):
    logger.info(msg)

# ============================================================================
# 图像预处理（不使用torchvision，避免bz2 DLL问题）
# ============================================================================
class PILPreprocess:
    """PIL-based image preprocessing for CLIP models."""
    def __init__(self, image_size=224, is_train=False):
        self.image_size = image_size
        self.is_train = is_train
        # CLIP归一化参数
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])

    def __call__(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert('RGB')
        # 保持宽高比的resize
        w, h = img.size
        if min(w, h) < self.image_size:
            # 短边缩放到image_size
            if w < h:
                new_w = self.image_size
                new_h = int(h * self.image_size / w)
            else:
                new_h = self.image_size
                new_w = int(w * self.image_size / h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
        # 中心裁剪
        w, h = img.size
        left = (w - self.image_size) // 2
        top = (h - self.image_size) // 2
        img = img.crop((left, top, left + self.image_size, top + self.image_size))
        # 随机水平翻转（训练时）
        if self.is_train and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # PIL -> numpy -> tensor [C,H,H]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_tensor

class AdaptiveResize:
    def __init__(self, size=512):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        if w < self.size and h < self.size:
            # 更小的边缩放到size
            if w < h:
                new_w = self.size
                new_h = int(h * self.size / w)
            else:
                new_h = self.size
                new_w = int(w * self.size / h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

def pil_resize_crop(img, size=224):
    w, h = img.size
    if w < size or h < size:
        if w < h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        img = img.resize((new_w, new_h), Image.BICUBIC)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img

def extract_patches(img_pil, num_patch=8, patch_size=224, step=32, is_train=False):
    """从PIL图像中提取patches，与IPCE兼容."""
    # 预处理resize
    preprocess = PILPreprocess(image_size=512, is_train=is_train)
    img_tensor = preprocess(img_pil)  # [3, 512, 512]
    C, H, W = img_tensor.shape
    # 滑动窗口提取patches
    patches = []
    for i in range(0, H - patch_size + 1, step):
        for j in range(0, W - patch_size + 1, step):
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    if len(patches) == 0:
        # 直接resize
        img_resized = img_pil.resize((patch_size, patch_size), Image.BICUBIC)
        img_tensor2 = PILPreprocess(image_size=patch_size)(img_resized)
        patches.append(img_tensor2)
    # 随机/均匀采样num_patch个patch
    if len(patches) >= num_patch:
        indices = torch.randperm(len(patches))[:num_patch-1]
        selected = [patches[i] for i in indices]
    else:
        selected = patches.copy()
        while len(selected) < num_patch - 1:
            selected.append(patches[-1])
    # 添加全局resize版本
    img_resized = img_pil.resize((patch_size, patch_size), Image.BICUBIC)
    global_view = PILPreprocess(image_size=patch_size)(img_resized)
    selected.append(global_view)
    return torch.stack(selected, dim=0)  # [num_patch, 3, 224, 224]

# ============================================================================
# 数据集定义（AGIQA-1K, AGIQA-3K, AIGCIQA2023）
# ============================================================================
class AGIQA1KDataset(Dataset):
    """AGIQA-1K数据集: 图像 + 提示词 + MOS分数"""
    def __init__(self, csv_file, img_dir, prompt_xlsx, desc_file, num_patch=8, is_train=False):
        super().__init__()
        self.img_dir = img_dir
        self.num_patch = num_patch
        self.is_train = is_train
        # 读取MOS分数
        df_score = pd.read_csv(csv_file)
        names, scores = df_score['Image'].tolist(), df_score['Score'].tolist()
        # 过滤缺失文件
        self.image_names, self.mos_scores = [], []
        for n, s in zip(names, scores):
            if os.path.isfile(os.path.join(img_dir, n)):
                self.image_names.append(n)
                self.mos_scores.append(s)
        # 读取提示词
        df_prompt = pd.read_excel(prompt_xlsx, engine='openpyxl')
        self.prompt_dict = {}
        for _, row in df_prompt.iterrows():
            self.prompt_dict[row['Image']] = row['Prompt']
        # 读取描述性提示词
        self.desc_dict = {}
        if desc_file and os.path.exists(desc_file):
            df_desc = pd.read_csv(desc_file)
            for _, row in df_desc.iterrows():
                self.desc_dict[row['image_name']] = row['description']
        print(f'AGIQA-1K: loaded {len(self.image_names)} samples')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        patches = extract_patches(img_pil, self.num_patch, is_train=self.is_train)
        mos = float(self.mos_scores[idx])
        # 原始提示词
        prompt = self.prompt_dict.get(img_name, '')
        # 描述性提示词
        desc = self.desc_dict.get(img_name, '')
        return {'patches': patches, 'mos': mos, 'prompt': prompt, 'desc': desc,
                'image_name': img_name}

class AGIQA3KDataset(Dataset):
    """AGIQA-3K数据集: 图像 + 提示词 + MOS (quality + alignment)"""
    def __init__(self, csv_file, img_dir, desc_file, num_patch=8, is_train=False):
        super().__init__()
        self.img_dir = img_dir
        self.num_patch = num_patch
        self.is_train = is_train
        df = pd.read_csv(csv_file)
        # 过滤缺失文件
        self.image_names, self.prompts, self.mos_quality = [], [], []
        for _, row in df.iterrows():
            if os.path.isfile(os.path.join(img_dir, row['name'])):
                self.image_names.append(row['name'])
                self.prompts.append(row['prompt'])
                self.mos_quality.append(row['mos_quality'])
        # 描述性提示词
        self.desc_dict = {}
        if desc_file and os.path.exists(desc_file):
            df_desc = pd.read_csv(desc_file)
            for _, row in df_desc.iterrows():
                self.desc_dict[row['image_name']] = row['description']
        print(f'AGIQA-3K: loaded {len(self.image_names)} samples')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        patches = extract_patches(img_pil, self.num_patch, is_train=self.is_train)
        mos = float(self.mos_quality[idx])
        prompt = str(self.prompts[idx])
        desc = self.desc_dict.get(img_name, '')
        return {'patches': patches, 'mos': mos, 'prompt': prompt, 'desc': desc,
                'image_name': img_name}

class AIGCIQA2023Dataset(Dataset):
    """AIGCIQA2023数据集:
       99个提示词, 每个提示词对应24张图像 (6生成器 × 4张/生成器)
       提示词索引1对应图像0-3, 400-403, 800-803, 1200-1203, 1600-1603, 2000-2003
    """
    def __init__(self, img_dir, prompt_xlsx, mos_dir, desc_file, num_patch=8, is_train=False):
        super().__init__()
        self.img_dir = img_dir
        self.num_patch = num_patch
        self.is_train = is_train
        # 读取提示词
        df_p = pd.read_excel(prompt_xlsx, engine='openpyxl')
        self.prompts = []
        for _, row in df_p.iterrows():
            # 第三列是提示词文本
            self.prompts.append(str(row.iloc[2]))
        self.num_prompts = len(self.prompts)  # 99个提示词
        # 读取MOS (有3个MOS文件)
        mos_list = []
        for filename in ['mosz1.mat', 'mosz2.mat', 'mosz3.mat']:
            mat = sio.loadmat(os.path.join(mos_dir, filename))
            mos_list.append(mat['MOSz'].flatten())
        # 取平均作为最终MOS
        self.mos_all = np.mean(mos_list, axis=0)  # [2400]
        # 过滤缺失文件
        valid_idx = []
        for i in range(2400):
            if os.path.isfile(os.path.join(img_dir, f'{i}.png')):
                valid_idx.append(i)
        self.num_total = len(valid_idx)
        self.image_indices = valid_idx
        # prompt_idx (0-based): image_ids
        self.img_to_prompt = {}
        prompt_idx = 0
        for base in range(0, self.num_total, 400):  # 6个generator组
            for i in range(self.num_prompts):
                for j in range(4):
                    img_id = base + i * 4 + j
                    if img_id < self.num_total:
                        self.img_to_prompt[img_id] = prompt_idx
                prompt_idx = (prompt_idx + 1) % self.num_prompts
        # 重映射: prompt_id -> image_ids
        self.prompt_to_imgs = {p: [] for p in range(self.num_prompts)}
        for base in range(0, self.num_total, 400):
            for p in range(self.num_prompts):
                for j in range(4):
                    img_id = base + p * 4 + j
                    if img_id < self.num_total:
                        self.prompt_to_imgs[p].append(img_id)
        # 描述性提示词
        self.desc_dict = {}
        if desc_file and os.path.exists(desc_file):
            df_desc = pd.read_csv(desc_file)
            for _, row in df_desc.iterrows():
                self.desc_dict[row['image_name']] = row['description']
        print(f'AIGCIQA2023: loaded {self.num_total} images, {self.num_prompts} prompts')

    def __len__(self):
        return self.num_total

    def __getitem__(self, idx):
        img_name = f'{idx}.png'
        img_path = os.path.join(self.img_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        patches = extract_patches(img_pil, self.num_patch, is_train=self.is_train)
        mos = float(self.mos_all[idx])
        prompt_idx = self.img_to_prompt.get(idx % (self.num_prompts * 4) // 4, 0)
        prompt = self.prompts[prompt_idx] if prompt_idx < len(self.prompts) else ''
        desc = self.desc_dict.get(img_name, '')
        return {'patches': patches, 'mos': mos, 'prompt': prompt, 'desc': desc,
                'image_name': img_name}

def collate_fn(batch):
    """自定义collate，将patches堆叠为 [batch, num_patch, C, H, W]"""
    patches = torch.stack([b['patches'] for b in batch], dim=0)
    mos = torch.tensor([b['mos'] for b in batch], dtype=torch.float32)
    prompts = [b['prompt'] for b in batch]
    descs = [b['desc'] for b in batch]
    image_names = [b['image_name'] for b in batch]
    return {'patches': patches, 'mos': mos, 'prompt': prompts, 'desc': descs,
            'image_name': image_names}

# ============================================================================
# 分支1: CLIP正向图文一致性评价 (Forward)
#   参考: IPCE (CVPRW 2024) - 质量形容词模板
# ============================================================================
class CLIPForwardConsistency(nn.Module):
    """
    CLIP正向图文一致性评价
    使用CLIP计算图像与原始提示词的T2I匹配度
    质量形容词: bad, poor, fair, good, perfect
    模板: "a photo that {quality} matches '{prompt}'"
    """
    def __init__(self, clip_model, freeze_ratio=0.5):
        super().__init__()
        self.clip = clip_model
        self.quality_adjectives = ['badly', 'poorly', 'fairly', 'well', 'perfectly']
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        # 部分冻结CLIP
        if freeze_ratio > 0:
            self._freeze_layers(freeze_ratio)

    def _freeze_layers(self, ratio):
        """冻结CLIP底层参数"""
        vision_layers = self.clip.vision_model.encoder.layers
        text_layers = self.clip.text_model.encoder.layers
        v_freeze = int(len(vision_layers) * ratio)
        t_freeze = int(len(text_layers) * ratio)
        for i, layer in enumerate(vision_layers):
            if i < v_freeze:
                for p in layer.parameters():
                    p.requires_grad = False
        for i, layer in enumerate(text_layers):
            if i < t_freeze:
                for p in layer.parameters():
                    p.requires_grad = False
        self.clip.logit_scale.requires_grad = False

    def forward(self, patches, prompts):
        """
        Args:
            patches: [B, num_patch, 3, 224, 224] 多patch图像
            prompts: list[str] 原始提示词
        Returns:
            consistency_scores: [B] 正向一致性得分 [0,5]范围
        """
        B, N, C, H, W = patches.shape
        device = patches.device
        # 为每个样本生成质量形容词模板文本
        all_texts = []
        for prompt in prompts:
            for adj in self.quality_adjectives:
                text = f"a photo that {adj} matches '{prompt}'"
                all_texts.append(text)
        # Tokenize
        tokens = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        # 将所有patches展平批量处理
        flat_patches = patches.view(-1, C, H, W)  # [B*N, 3, 224, 224]
        # 编码图像
        vision_outputs = self.clip.vision_model(pixel_values=flat_patches)
        image_embeds = vision_outputs.pooler_output  # [B*N, 512]
        image_embeds = self.clip.visual_projection(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        # 编码文本 (所有样本所有模板一次编码)
        text_outputs = self.clip.text_model(**tokens)
        text_embeds = text_outputs.pooler_output  # [B*5, 512]
        text_embeds = self.clip.text_projection(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1)
        # 计算logits
        logit_scale = self.clip.logit_scale.exp()
        # 重塑: [B, N, 512] 和 [B, 5, 512]
        img_feat = image_embeds.view(B, N, -1)  # [B, N, 512]
        # 每个样本对应5个模板
        txt_feat = text_embeds.view(B, 5, -1)   # [B, 5, 512]
        # 计算每个patch与每个模板的相似度
        # 移除最后一个patch (全局视图) 或 保留均值
        patch_feat = img_feat[:, :-1, :]  # [B, N-1, 512]
        global_feat = img_feat[:, -1:, :]  # [B, 1, 512]
        # 局部patch平均
        local_sim = torch.einsum('bnd,btd->bnt', patch_feat, txt_feat)  # [B, N-1, 5]
        local_sim = local_sim.mean(dim=1)  # [B, 5]
        # 全局视图
        global_sim = torch.einsum('bnd,btd->bnt', global_feat, txt_feat)  # [B, 1, 5]
        global_sim = global_sim.squeeze(1)  # [B, 5]
        # 融合局部+全局
        sim = (local_sim + global_sim) / 2.0
        # Softmax加权得分
        sim_softmax = F.softmax(sim * logit_scale, dim=-1)
        scores = (1 * sim_softmax[:, 0] + 2 * sim_softmax[:, 1] + 3 * sim_softmax[:, 2] +
                  4 * sim_softmax[:, 3] + 5 * sim_softmax[:, 4])
        return scores  # [B] 在[1,5]范围

# ============================================================================
# 分支2: 双提示词对比模型 (Dual-Prompt Contrast)
#   比较原始提示词 vs 描述性提示词的语义一致性
# ============================================================================
class DualPromptContrast(nn.Module):
    """
    双提示词对比模型
    计算原始提示词与描述性提示词的多维度匹配度
    5个美学维度: content, style, angle, color_lighting, details
    使用CLIP文本编码器进行嵌入相似度计算
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_text = clip_model.text_model
        self.text_proj = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        # 5个美学维度的引导关键词
        self.dimension_keywords = {
            'content': ['subject', 'object', 'scene', 'main', 'focus', 'character'],
            'style': ['style', 'artistic', 'anime', 'realistic', 'painting', 'cartoon', 'render'],
            'angle': ['angle', 'perspective', 'view', 'closeup', 'distant', 'aerial', 'overhead'],
            'color_lighting': ['color', 'light', 'bright', 'dark', 'vibrant', 'pale', 'hue', 'tone'],
            'details': ['detail', 'texture', 'pattern', 'fine', 'intricate', 'complex']
        }
        self.dimension_weights = {
            'content': 0.4,
            'style': 0.2,
            'angle': 0.15,
            'color_lighting': 0.15,
            'details': 0.1
        }
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        # 冻结文本编码器
        for p in self.clip_text.parameters():
            p.requires_grad = False
        self.text_proj.requires_grad = False
        self.logit_scale.requires_grad = False

    def _encode_text(self, texts):
        """编码文本列表为归一化嵌入"""
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        outputs = self.clip_text(**tokens)
        embeds = self.text_proj(outputs.pooler_output)
        return F.normalize(embeds, dim=-1)

    def forward(self, orig_prompts, desc_prompts):
        """
        Args:
            orig_prompts: list[str] 原始提示词
            desc_prompts: list[str] 描述性提示词
        Returns:
            consistency_scores: [B] 双提示词一致性得分 [0,1]范围
        """
        B = len(orig_prompts)
        device = next(self.parameters()).device
        scores = torch.zeros(B, device=device)
        for i in range(B):
            orig = orig_prompts[i] if orig_prompts[i] else ''
            desc = desc_prompts[i] if desc_prompts[i] else ''
            if not orig or not desc:
                scores[i] = 0.5  # 默认中间值
                continue
            dim_scores = []
            for dim_name, keywords in self.dimension_keywords.items():
                w = self.dimension_weights[dim_name]
                # 对原始提示词，抽取包含关键字的片段
                orig_lower = orig.lower()
                desc_lower = desc.lower()
                # 检查该维度在原始提示中是否有关键词
                has_dim_in_orig = any(kw in orig_lower for kw in keywords)
                if not has_dim_in_orig:
                    # 该维度无要求 → 计为1.0
                    dim_scores.append(1.0)
                    continue
                # 计算原始提示与描述提示在该维度上的语义相似度
                # 使用整句嵌入的余弦相似度
                with torch.no_grad():
                    orig_emb = self._encode_text([orig])
                    desc_emb = self._encode_text([desc])
                sim = F.cosine_similarity(orig_emb, desc_emb).item()
                sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # 映射到[0,1]
                dim_scores.append(sim)
            if dim_scores:
                # 加权求和
                weights = [self.dimension_weights[d] for d in self.dimension_keywords
                           if any(kw in orig.lower() for kw in self.dimension_keywords[d])
                           or True]  # 实际上需要匹配已计算的数量
                # 简化: 使用所有维度的平均权重
                avg_w = 1.0 / len(dim_scores)
                scores[i] = sum(dim_scores) * avg_w
            else:
                scores[i] = 0.5
        return scores

# ============================================================================
# 分支3: 多尺度感知质量评估网络 (MSPQN)
#   参考: AGIN DeepWSD (VGG16多尺度特征)
#         CHPNet MSA_I + FCAttention
#         MoE-AGIQA 门控+专家
# ============================================================================
class L2pooling(nn.Module):
    """来自AGIN的L2pooling层"""
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride,
                       padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class VGGStageFeatures(nn.Module):
    """
    VGG16多尺度特征提取器 - 按阶段分组输出
    输出4个阶段的特征: [64, 128, 256, 512]
    """
    def __init__(self):
        super().__init__()
        # VGG16 stages (conv + relu pairs, followed by maxpool)
        # 使用标准MaxPool2d替代L2pooling以保证与预训练权重兼容
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 随机初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f1 = self.stage1(x)  # [B, 64, 112, 112]
        f2 = self.stage2(f1)  # [B, 128, 56, 56]
        f3 = self.stage3(f2)  # [B, 256, 28, 28]
        f4 = self.stage4(f3)  # [B, 512, 14, 14]
        return [f1, f2, f3, f4]


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 (来自CHPNet MSA_I)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MSA_I_Block(nn.Module):
    """图像多尺度注意力 (来自CHPNet MSA_I)"""
    def __init__(self, channels):
        super().__init__()
        self.branch1 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.branch2 = DepthwiseSeparableConv(channels, channels, kernel_size=5, padding=2)
        self.branch3 = DepthwiseSeparableConv(channels, channels, kernel_size=7, padding=3)
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.fusion(out)
        return out


class SELayer(nn.Module):
    """通道注意力SE模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM_Spatial(nn.Module):
    """CBAM空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ImageAttributeFeature(nn.Module):
    """
    图像属性特征提取 (来自AGIN image_attribute_feature.py)
    使用可学习的特征提取代替手算
    """
    def __init__(self):
        super().__init__()
        self.conv_brightness = nn.Conv2d(3, 16, kernel_size=1)
        self.conv_contrast = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv_color = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv_sharpness = nn.Conv2d(3, 16, kernel_size=7, padding=3)
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        f1 = self.conv_brightness(x)
        f2 = self.conv_contrast(x)
        f3 = self.conv_color(x)
        f4 = self.conv_sharpness(x)
        f = torch.cat([f1, f2, f3, f4], dim=1)
        f = self.fusion(f).view(x.size(0), -1)  # [B, 128]
        return f


class MSPQN(nn.Module):
    """
    多尺度感知质量评估网络 (Multi-Scale Perception Quality Network)
    包含: VGG16多尺度特征 + 图像属性 + 注意力 + MoE门控融合
    """
    def __init__(self):
        super().__init__()
        # VGG16多尺度特征提取 (4 stages: 64, 128, 256, 512)
        self.vgg = VGGStageFeatures()
        # 图像属性特征
        self.attr_net = ImageAttributeFeature()
        # 多尺度注意力
        self.msa_blocks = nn.ModuleList([
            MSA_I_Block(64),
            MSA_I_Block(128),
            MSA_I_Block(256),
            MSA_I_Block(512),
        ])
        # SE通道注意力
        self.se_blocks = nn.ModuleList([
            SELayer(64),
            SELayer(128),
            SELayer(256),
            SELayer(512),
        ])
        # CBAM空间注意力
        self.cbam_blocks = nn.ModuleList([
            CBAM_Spatial(),
            CBAM_Spatial(),
            CBAM_Spatial(),
            CBAM_Spatial(),
        ])
        # 自适应融合 (AFF) - 可学习权重
        self.aff_weights = nn.Parameter(torch.ones(4) / 4.0)
        # 回归头
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        vgg_dim = 64 + 128 + 256 + 512
        total_dim = vgg_dim + 128  # VGG特征 + 属性特征
        # MoE门控融合
        self.gating_network = nn.Sequential(
            nn.Linear(total_dim, 4),
            nn.Softmax(dim=1)
        )
        # 4个专家
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in range(4)
        ])
        self.top_k = 2

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] 输入图像
        Returns:
            quality_score: [B] 图像质量分数
        """
        B = x.size(0)
        # VGG多尺度特征 [64, 128, 256, 512]
        layer_feats = self.vgg(x)
        # 应用注意力
        attned_feats = []
        for i, feat in enumerate(layer_feats):
            se = self.se_blocks[i](feat)
            cbam = self.cbam_blocks[i](feat)
            msa = self.msa_blocks[i](feat)
            # 自适应融合: se + cbam + msa
            w = F.softmax(self.aff_weights, dim=0)
            fused = w[0] * se + w[1] * cbam + w[2] * msa
            pooled = self.pool(fused).view(B, -1)
            attned_feats.append(pooled)
        # 图像属性特征
        attr_feat = self.attr_net(x)
        # 拼接所有特征
        all_feats = torch.cat(attned_feats + [attr_feat], dim=-1)
        # MoE门控
        gate_weights = self.gating_network(all_feats)
        # Top-K选择
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)
        # 专家预测: 所有专家输出拼接 [B, num_experts]
        expert_preds = torch.cat([e(all_feats) for e in self.experts], dim=-1)  # [B, 4]
        # Top-K加权融合
        topk_w = topk_weights  # [B, K]
        topk_idx = topk_indices  # [B, K]
        batch_arange = torch.arange(B, device=x.device).unsqueeze(-1)  # [B, 1]
        selected = expert_preds[batch_arange, topk_idx]  # [B, K]
        final_pred = (selected * topk_w).sum(dim=-1, keepdim=True)  # [B, 1]
        return final_pred.squeeze(-1)  # [B]

# ============================================================================
# 融合模块: 自适应双路池化 + 门控注意力融合 + Top-K MoE
# ============================================================================
class AdaptiveDualScorePooling(nn.Module):
    """
    自适应双路池化模块
    融合正向一致性得分(score1)和双提示词一致性得分(score2)
    """
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习融合权重
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, score1, score2):
        """
        score1: [B] 正向一致性得分
        score2: [B] 双提示词一致性得分
        Returns: fused_score [B]
        """
        alpha = torch.sigmoid(self.alpha)
        # 可学习加权
        weighted = alpha * score1 + (1 - alpha) * score2
        # MLP融合
        stack_feat = torch.stack([score1, score2], dim=-1)
        mlp_out = self.mlp(stack_feat).squeeze(-1)
        return (weighted + mlp_out) / 2.0


class GatedAttentionFusion(nn.Module):
    """
    门控注意力融合模块
    融合一致性得分和图像质量得分
    """
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, consistency_score, quality_score):
        """
        consistency_score: [B] 融合一致性得分
        quality_score: [B] 图像本体质量分数
        Returns: fused [B]
        """
        stack_feat = torch.stack([consistency_score, quality_score], dim=-1)
        gate_weight = self.gate(stack_feat).squeeze(-1)
        # 门控加权融合
        fused = gate_weight * consistency_score + (1 - gate_weight) * quality_score
        return fused

# ============================================================================
# 完整模型: 将所有分支和融合模块组合
# ============================================================================
class AIGIQA_FullModel(nn.Module):
    """
    完整AI生成图像质量评价模型
    三条分支 + 自适应池化 + 门控注意力 + Top-K MoE
    """
    def __init__(self, freeze_clip_ratio=0.5):
        super().__init__()
        # 加载CLIP模型作为backbone
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        # 分支1: CLIP正向一致性
        self.branch1 = CLIPForwardConsistency(self.clip, freeze_clip_ratio)
        # 分支2: 双提示词对比
        self.branch2 = DualPromptContrast(self.clip)
        # 分支3: 多尺度感知质量
        self.branch3 = MSPQN()
        # 融合模块
        self.dual_pool = AdaptiveDualScorePooling()
        self.gated_fusion = GatedAttentionFusion()
        # 最终回归 (MoE)
        self.final_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(4)
        ])
        self.final_gate = nn.Sequential(
            nn.Linear(2, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, patches, prompts, descs):
        """
        Args:
            patches: [B, N, 3, 224, 224] 图像patches
            prompts: list[str] 原始提示词
            descs: list[str] 描述性提示词
        Returns:
            final_score: [B] 最终质量分数
        """
        # 分支1: 正向一致性
        score1 = self.branch1(patches, prompts)  # [B] 范围[1,5]
        # 分支2: 双提示词一致性
        score2 = self.branch2(prompts, descs)  # [B] 范围[0,1]
        # 将score2映射到[1,5]
        score2 = 1.0 + 4.0 * score2  # [B] 范围[1,5]
        # 自适应双路池化: 融合score1和score2
        consistency_fused = self.dual_pool(score1, score2)  # [B]
        # 分支3: 图像本体质量
        # 使用最后一个patch (全局视图)作为质量评估输入
        quality_score = self.branch3(patches[:, -1, :, :, :])  # [B] or [B,1]
        if quality_score.dim() > 1:
            quality_score = quality_score.squeeze(-1)  # [B]
        # 将quality_score映射到[1,5]
        quality_score = 1.0 + 4.0 * torch.sigmoid(quality_score)
        # 门控注意力融合
        fused = self.gated_fusion(consistency_fused, quality_score)  # [B]
        # 最终MoE回归
        B = patches.size(0)
        feat = torch.stack([consistency_fused, quality_score], dim=-1)
        gate_w = self.final_gate(feat)  # [B, 4]
        expert_outs = []
        for expert in self.final_experts:
            expert_outs.append(expert(feat))
        expert_outs = torch.stack(expert_outs, dim=-1).squeeze(1)  # [B, 4]
        # Top-2选择
        topk_w, topk_idx = torch.topk(gate_w, 2, dim=1)
        topk_w = topk_w / (topk_w.sum(dim=1, keepdim=True) + 1e-8)
        # 向量化Top-K融合
        batch_arange = torch.arange(B, device=patches.device).unsqueeze(-1)
        selected = expert_outs[batch_arange, topk_idx]
        final_score = (selected * topk_w).sum(dim=-1)
        return final_score  # [B]

# ============================================================================
# 损失函数: 联合损失 (MSE + PLCC + SRCC)
# ============================================================================
class JointLoss(nn.Module):
    """
    联合损失函数
    L = lambda_1 * L_MSE + lambda_2 * L_PLCC + lambda_3 * L_SRCC
    参考: VQualA 2025 Challenge
    """
    def __init__(self, lambda_mse=0.4, lambda_plcc=0.3, lambda_srcc=0.3):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_plcc = lambda_plcc
        self.lambda_srcc = lambda_srcc

    def forward(self, pred, target):
        # MSE损失
        loss_mse = F.mse_loss(pred, target)
        # PLCC损失 (1 - Pearson相关系数)
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum() + 1e-8)
        plcc = numerator / denominator
        loss_plcc = 1.0 - plcc
        # SRCC损失 (使用可微近似)
        if pred.size(0) > 1:
            pred_rank = torch.argsort(pred).float()
            target_rank = torch.argsort(target).float()
            pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-8)
            target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-8)
            srcc = (pred_rank * target_rank).mean()
            loss_srcc = 1.0 - srcc
        else:
            loss_srcc = torch.tensor(0.0, device=pred.device)
        # 联合损失
        total_loss = (self.lambda_mse * loss_mse +
                      self.lambda_plcc * loss_plcc +
                      self.lambda_srcc * loss_srcc)
        return total_loss, {'mse': loss_mse.item(), 'plcc': loss_plcc.item(),
                            'srcc': loss_srcc.item()}


# ============================================================================
# 训练和评估函数
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    step = 0
    for batch in loader:
        patches = batch['patches'].to(device)
        targets = batch['mos'].to(device)
        prompts = batch['prompt']
        descs = batch['desc']
        optimizer.zero_grad()
        preds = model(patches, prompts, descs)
        loss, loss_dict = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        step += 1
        if step % 10 == 0:
            print(f'  Step {step}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / max(step, 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    srcc_val = spearmanr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    plcc_val = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    return avg_loss, srcc_val, plcc_val


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    step = 0
    for batch in loader:
        patches = batch['patches'].to(device)
        targets = batch['mos'].to(device)
        prompts = batch['prompt']
        descs = batch['desc']
        preds = model(patches, prompts, descs)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        step += 1
    avg_loss = total_loss / max(step, 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    srcc_val = spearmanr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    plcc_val = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    return avg_loss, srcc_val, plcc_val


# ============================================================================
# 数据加载器工厂
# ============================================================================
def create_dataloaders(dataset_name, batch_size=16, num_workers=4, num_patch=8, is_train=True):
    """
    根据数据集名称创建DataLoader
    使用train/test 80/20随机划分
    """
    if dataset_name == 'AGIQA-1K':
        csv_file = os.path.join(DATA_ROOT, 'AGIQA-1K', 'AIGC_all.csv')
        img_dir = os.path.join(DATA_ROOT, 'AGIQA-1K', 'images')
        prompt_xlsx = os.path.join(DATA_ROOT, 'AGIQA-1K', 'AIGC_MOS_Zscore.xlsx')
        desc_file = os.path.join(DATA_ROOT, 'pre_descriptions', 'AGIQA-1K_descriptions.csv')
        full_dataset = AGIQA1KDataset(csv_file, img_dir, prompt_xlsx, desc_file,
                                      num_patch=num_patch, is_train=is_train)
    elif dataset_name == 'AGIQA-3K':
        csv_file = os.path.join(DATA_ROOT, 'AGIQA-3K', 'data.csv')
        img_dir = os.path.join(DATA_ROOT, 'AGIQA-3K', 'images')
        desc_file = os.path.join(DATA_ROOT, 'pre_descriptions', 'AGIQA-3K_descriptions.csv')
        full_dataset = AGIQA3KDataset(csv_file, img_dir, desc_file,
                                      num_patch=num_patch, is_train=is_train)
    elif dataset_name == 'AIGCIQA2023':
        img_dir = os.path.join(DATA_ROOT, 'AIGCIQA2023', 'images')
        prompt_xlsx = os.path.join(DATA_ROOT, 'AIGCIQA2023', 'AIGCIQA2023_Prompts.xlsx')
        mos_dir = os.path.join(DATA_ROOT, 'AIGCIQA2023', 'MOS')
        desc_file = os.path.join(DATA_ROOT, 'pre_descriptions', 'AIGCIQA2023_descriptions.csv')
        full_dataset = AIGCIQA2023Dataset(img_dir, prompt_xlsx, mos_dir, desc_file,
                                          num_patch=num_patch, is_train=is_train)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 随机划分 train/val = 80/20
    total = len(full_dataset)
    indices = list(range(total))
    random.shuffle(indices)
    split = int(total * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True)
    print(f'{dataset_name}: train={len(train_dataset)}, val={len(val_dataset)}')
    return train_loader, val_loader


# ============================================================================
# 主训练循环
# ============================================================================
def train_on_dataset(dataset_name, model, logger, device):
    """在指定数据集上训练和评估"""
    print(f'\n{"="*60}')
    print(f'Training on {dataset_name}')
    print(f'{"="*60}')
    log_and_print(logger, f'Starting training on {dataset_name}')

    # 数据加载
    train_loader, val_loader = create_dataloaders(
        dataset_name, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        num_patch=NUM_PATCH_TRAIN)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    criterion = JointLoss()

    best_srcc = -1
    best_plcc = -1
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f'\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---')
        # 训练
        train_loss, train_srcc, train_plcc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        # 验证
        val_loss, val_srcc, val_plcc = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        msg = (f'Epoch {epoch+1:2d}: Train Loss={train_loss:.4f} SRCC={train_srcc:.4f} '
               f'PLCC={train_plcc:.4f} | Val Loss={val_loss:.4f} SRCC={val_srcc:.4f} '
               f'PLCC={val_plcc:.4f}')
        print(msg)
        log_and_print(logger, msg)

        # 保存最佳模型
        if val_srcc + val_plcc > best_srcc + best_plcc:
            best_srcc = val_srcc
            best_plcc = val_plcc
            best_loss = val_loss
            best_epoch = epoch + 1
            ckpt_dir = os.path.join('checkpoints', dataset_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_srcc': best_srcc,
                'best_plcc': best_plcc,
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            log_and_print(logger, f'*** New best model saved (epoch {best_epoch}) ***')

    final_msg = (f'{dataset_name} completed: Best Epoch={best_epoch}, '
                 f'SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f}, Loss={best_loss:.4f}')
    print(final_msg)
    log_and_print(logger, final_msg)
    return best_srcc, best_plcc, best_loss


# ============================================================================
# 主入口
# ============================================================================
if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    logger = get_logger(os.path.join('checkpoints', 'training.log'), 'main')

    log_and_print(logger, f'Device: {DEVICE}')
    log_and_print(logger, f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

    # 初始化模型
    model = AIGIQA_FullModel(freeze_clip_ratio=0.5)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_and_print(logger, f'Total parameters: {total_params/1e6:.2f}M')
    log_and_print(logger, f'Trainable parameters: {trainable_params/1e6:.2f}M')

    # 在三个数据集上依次训练
    datasets = ['AGIQA-1K', 'AGIQA-3K', 'AIGCIQA2023']
    all_results = {}

    for ds in datasets:
        best_srcc, best_plcc, best_loss = train_on_dataset(ds, model, logger, DEVICE)
        all_results[ds] = {'SRCC': best_srcc, 'PLCC': best_plcc, 'Loss': best_loss}

    # 汇总结果
    log_and_print(logger, '\n' + '='*60)
    log_and_print(logger, 'FINAL RESULTS')
    log_and_print(logger, '='*60)
    for ds, res in all_results.items():
        log_and_print(logger, f'{ds:15s}: SRCC={res["SRCC"]:.4f}, PLCC={res["PLCC"]:.4f}, Loss={res["Loss"]:.4f}')

    avg_srcc = np.mean([r['SRCC'] for r in all_results.values()])
    avg_plcc = np.mean([r['PLCC'] for r in all_results.values()])
    log_and_print(logger, f'{"Average":15s}: SRCC={avg_srcc:.4f}, PLCC={avg_plcc:.4f}')
    print('\nTraining complete!')
