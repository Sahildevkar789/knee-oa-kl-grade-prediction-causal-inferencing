"""Shared pieces for KL-grade classification: data loading, models, and evaluation."""
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision.models import (efficientnet_b0, EfficientNet_B0_Weights,
                                resnet50, ResNet50_Weights,
                                densenet121, DenseNet121_Weights,
                                convnext_tiny, ConvNeXt_Tiny_Weights)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224          # classifier input size
SEG_SIZE = 256          # size the segmentation models were trained at
GRADES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
KL_TEXT = {0: "No osteoarthritis", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"}
SEG_CKPTS = {"unet": "unet_best.pt", "unetpp": "unetpp_best.pt"}
BACKBONES = ["efficientnet_b0", "resnet50", "densenet121", "convnext_tiny"]


def run_name(mask_source, backbone):
    """Backbone is always part of the name so new runs never overwrite the original
    kl_baseline/kl_unet/kl_unetpp results from the efficientnet_b0-only comparison."""
    base = f"kl_{mask_source}" if mask_source in SEG_CKPTS else "kl_baseline"
    return f"{base}_{backbone}"


def list_split(data_root, split):
    """Returns image paths and KL labels from data_root/split/<grade>/*."""
    paths, labels = [], []
    for g in range(5):
        files = sorted(glob.glob(os.path.join(data_root, split, str(g), "*")))
        paths += files
        labels += [g] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found in {os.path.join(data_root, split)} - check --data-root")
    return paths, labels


def get_transforms():
    train_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5),
                          A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.Normalize(MEAN, STD), ToTensorV2()])
    eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()])
    return train_tf, eval_tf


def apply_clahe(img_rgb):
    """CLAHE (clipLimit 2.0, tile 8x8) on the grayscale version of an RGB image, then back to 3 channels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class KLDataset(Dataset):
    def __init__(self, paths, labels, transform, use_clahe=False):
        self.paths, self.labels, self.transform, self.use_clahe = paths, labels, transform, use_clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_clahe:
            img = apply_clahe(img)
        return self.transform(image=img)["image"], self.labels[i]


def _expand_to_4ch(old_conv):
    """Builds a 4-input-channel copy of a conv layer: first 3 channels keep the pretrained
    weights, the 4th (mask channel) starts as their average."""
    new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding,
                         bias=old_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias
    return new_conv


class CBAM(nn.Module):
    """Channel + spatial attention (Woo et al. 2018), applied to a conv feature map."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        ch_att = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ch_att
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * sp_att, sp_att


class ConvNeXtCBAM(nn.Module):
    """Wraps a convnext_tiny model with CBAM on its final feature map, before pooling/classifier."""
    def __init__(self, base, channels):
        super().__init__()
        self.features = base.features
        self.cbam = CBAM(channels)
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.last_attn = None   # most recent spatial attention map, for visualization

    def forward(self, x):
        x = self.features(x)
        x, attn = self.cbam(x)
        self.last_attn = attn.detach()
        x = self.avgpool(x)
        return self.classifier(x)


class ConvNeXtCBAMTransformer(nn.Module):
    """Hybrid 1: ConvNeXt-Tiny+CBAM's final feature map (768x7x7) -> projected to 256-dim
    tokens -> CLS token + learnable positional embeddings -> 2-layer Transformer encoder
    (4 heads, dropout 0.1) -> CLS output -> CORAL head (4 thresholds)."""
    def __init__(self, base_convnext, cbam_channels, proj_dim=256, n_tokens=49,
                n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.features = base_convnext.features
        self.cbam = CBAM(cbam_channels)
        self.proj = nn.Conv2d(cbam_channels, proj_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, proj_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=n_heads,
                                                    dim_feedforward=proj_dim * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(proj_dim, 4)
        self.last_attn = None   # CBAM spatial attention, kept for visualization parity with ConvNeXtCBAM

    def forward(self, x):
        x = self.features(x)
        x, attn = self.cbam(x)
        self.last_attn = attn.detach()
        x = self.proj(x)
        b = x.shape[0]
        tokens = x.flatten(2).transpose(1, 2)               # (B, 49, 256)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)   # (B, 50, 256)
        tokens = tokens + self.pos_embed
        out = self.transformer(tokens)
        return self.head(out[:, 0])


def build_hybrid1(ckpt_dir, device, init_from="kl_final_best.pt"):
    """Builds Hybrid 1, initializing its features/CBAM weights from a trained ConvNeXtCBAM
    checkpoint (e.g. kl_final) by copying any state_dict keys with matching names+shapes -
    the new projection/CLS/positional/transformer/head layers have no match and stay at their
    default random init."""
    base = convnext_tiny(weights=None)
    in_f = base.classifier[2].in_features
    model = ConvNeXtCBAMTransformer(base, cbam_channels=in_f)
    if init_from:
        src_state = torch.load(os.path.join(ckpt_dir, init_from), map_location=device)
        model_state = model.state_dict()
        n_loaded = 0
        for k, v in src_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                n_loaded += 1
        model.load_state_dict(model_state)
        print(f"Hybrid1: initialized {n_loaded} tensors from {init_from} (features.*/cbam.*)")
    return model.to(device)


class ConvNeXtCBAMFusion(nn.Module):
    """Hybrid 2: pooled ConvNeXt-Tiny+CBAM image features concatenated with a small MLP over
    tabular features (age/BMI, standardized by the caller), followed by a CORAL head."""
    def __init__(self, base_convnext, cbam_channels, tabular_dim, tab_hidden=32, img_proj=128):
        super().__init__()
        self.features = base_convnext.features
        self.cbam = CBAM(cbam_channels)
        self.avgpool = base_convnext.avgpool
        self.img_proj = nn.Linear(cbam_channels, img_proj)
        self.tab_mlp = nn.Sequential(nn.Linear(tabular_dim, tab_hidden), nn.ReLU(inplace=True),
                                     nn.Linear(tab_hidden, tab_hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(img_proj + tab_hidden, 4)
        self.last_attn = None

    def forward(self, x, tab):
        x = self.features(x)
        x, attn = self.cbam(x)
        self.last_attn = attn.detach()
        x = self.avgpool(x).flatten(1)
        x = F.relu(self.img_proj(x))
        t = self.tab_mlp(tab)
        return self.head(torch.cat([x, t], dim=1))


def build_hybrid2(ckpt_dir, device, tabular_dim, init_from="kl_final_best.pt"):
    """Builds Hybrid 2, initializing its features/CBAM weights from a trained ConvNeXtCBAM
    checkpoint the same way build_hybrid1 does."""
    base = convnext_tiny(weights=None)
    in_f = base.classifier[2].in_features
    model = ConvNeXtCBAMFusion(base, cbam_channels=in_f, tabular_dim=tabular_dim)
    if init_from:
        src_state = torch.load(os.path.join(ckpt_dir, init_from), map_location=device)
        model_state = model.state_dict()
        n_loaded = 0
        for k, v in src_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                n_loaded += 1
        model.load_state_dict(model_state)
        print(f"Hybrid2: initialized {n_loaded} tensors from {init_from} (features.*/cbam.*)")
    return model.to(device)


class KLDatasetFusion(Dataset):
    """Like KLDataset, but also returns a tabular feature vector per image."""
    def __init__(self, paths, labels, tabular, transform, use_clahe=False):
        self.paths, self.labels, self.tabular = paths, labels, tabular
        self.transform, self.use_clahe = transform, use_clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_clahe:
            img = apply_clahe(img)
        img_t = self.transform(image=img)["image"]
        tab_t = torch.tensor(self.tabular[i], dtype=torch.float32)
        return img_t, tab_t, self.labels[i]


def build_classifier(backbone, in_ch, pretrained=True, out_dim=5, attention=None):
    """ImageNet-pretrained backbone with a classification head. in_ch=4 adds a channel for the
    bone mask. out_dim=4 switches to a CORAL ordinal head (see coral_loss/coral_decode).
    attention="cbam" wraps convnext_tiny's final feature map with a CBAM block (unsupported
    for other backbones - this project only asked for it on convnext_tiny)."""
    if backbone == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_ch == 4:
            m.features[0][0] = _expand_to_4ch(m.features[0][0])
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, out_dim)
    elif backbone == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if in_ch == 4:
            m.conv1 = _expand_to_4ch(m.conv1)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
    elif backbone == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_ch == 4:
            m.features.conv0 = _expand_to_4ch(m.features.conv0)
        m.classifier = nn.Linear(m.classifier.in_features, out_dim)
    elif backbone == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_ch == 4:
            m.features[0][0] = _expand_to_4ch(m.features[0][0])
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, out_dim)
        if attention == "cbam":
            m = ConvNeXtCBAM(m, channels=in_f)
    else:
        raise ValueError(f"Unknown backbone: {backbone} (choices: {BACKBONES})")
    if attention == "cbam" and backbone != "convnext_tiny":
        raise ValueError("CBAM is only wired up for convnext_tiny in this project")
    return m


def compute_class_weights(train_y, mode, device):
    """mode: 'inverse' (default, matches the original recipe), 'sqrt', or 'none'."""
    counts = np.bincount(train_y, minlength=5)
    if mode == "none":
        return torch.ones(5, dtype=torch.float32, device=device)
    inv = len(train_y) / (5 * counts)
    if mode == "sqrt":
        inv = np.sqrt(inv)
    return torch.tensor(inv, dtype=torch.float32, device=device)


def coral_targets(y, num_thresholds=4):
    """y: (N,) ints 0-4 -> (N, num_thresholds) binary targets, target[:,k] = 1 if y > k."""
    thresholds = torch.arange(num_thresholds, device=y.device)
    return (y.unsqueeze(1) > thresholds).float()


def coral_pos_weights(train_y, device, mode="inverse", num_thresholds=4):
    """Per-threshold BCE pos_weight = n_negative / n_positive, from the training label distribution.
    mode mirrors compute_class_weights: 'inverse' (default) uses the ratio as-is, 'sqrt' takes its
    square root (gentler weighting), 'none' disables weighting (all 1s)."""
    y = torch.tensor(train_y)
    targets = coral_targets(y, num_thresholds)
    pos = targets.sum(dim=0).clamp(min=1)
    neg = (len(y) - targets.sum(dim=0)).clamp(min=1)
    w = neg / pos
    if mode == "none":
        w = torch.ones_like(w)
    elif mode == "sqrt":
        w = torch.sqrt(w)
    return w.to(device)


def coral_loss(logits, y, pos_weight):
    """4 independent binary 'grade > k' losses, each weighted for its own class imbalance."""
    targets = coral_targets(y, logits.shape[1])
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def coral_probs_to_grade(probs):
    """probs: (N,4) sigmoid outputs -> (N,) predicted grade = count of thresholds exceeded."""
    return (probs > 0.5).sum(dim=1)


def load_seg_model(kind, ckpt_dir, device):
    """Loads a trained U-Net or U-Net++ (from the Colab runs), frozen."""
    cls = smp.Unet if kind == "unet" else smp.UnetPlusPlus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    path = os.path.join(ckpt_dir, SEG_CKPTS[kind])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - download it from Google Drive into {ckpt_dir}")
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def add_mask_channel(imgs, seg_model):
    """Adds the segmentation model's bone-probability map as a 4th input channel."""
    if seg_model is None:
        return imgs
    with torch.no_grad():
        x = F.interpolate(imgs, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(seg_model(x).float())
        prob = F.interpolate(prob, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([imgs, prob.to(imgs.dtype)], dim=1)


def _forward_probs(model, imgs, seg_model, device, use_amp, loss_type):
    """One forward pass -> (N, 5) softmax [ce] or (N, 4) sigmoid-threshold-probs [coral]."""
    x = add_mask_channel(imgs.to(device), seg_model)
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x)
    if loss_type == "coral":
        return torch.sigmoid(logits.float())
    return torch.softmax(logits.float(), dim=1)


def run_eval(model, loader, seg_model, device, use_amp, loss_type="ce", tta=False):
    """Returns predicted grades and true grades for a whole loader."""
    preds, labels, _ = run_eval_probs(model, loader, seg_model, device, use_amp, loss_type, tta)
    return preds, labels


def run_eval_probs(model, loader, seg_model, device, use_amp, loss_type="ce", tta=False):
    """Returns predicted grades, true grades, and the raw per-threshold/per-class probabilities
    (5-dim softmax for loss_type="ce", 4-dim sigmoid "P(grade>k)" for loss_type="coral").
    tta=True averages the probabilities of each image with its horizontal flip."""
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for imgs, y in loader:
            p = _forward_probs(model, imgs, seg_model, device, use_amp, loss_type)
            if tta:
                flipped = torch.flip(imgs, dims=[3])
                p = (p + _forward_probs(model, flipped, seg_model, device, use_amp, loss_type)) / 2
            pred = coral_probs_to_grade(p) if loss_type == "coral" else p.argmax(1)
            preds.append(pred.cpu())
            labels.append(y)
            probs.append(p.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy()
