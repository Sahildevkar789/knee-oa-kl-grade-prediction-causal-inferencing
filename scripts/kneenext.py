"""KneeNeXt: a knee-specific modification of kl_final's ConvNeXt-Tiny classifier.

Clinical motivation (overall): kl_final looks at the whole 224x224 joint crop with one
global view and lets a generic CBAM attention block learn where to look on its own, purely
from label supervision. Radiologists grading KL severity instead work from specific,
known-in-advance anatomical evidence: the joint space itself (narrowing, best seen zoomed in
on the joint line) and asymmetric wear between the medial and lateral compartments (a knee
that has worn unevenly looks different left-vs-right in a way a symmetric "average" feature
map destroys by pooling). KneeNeXt adds four optional modules on top of kl_final's backbone,
each targeting one piece of that domain knowledge, so they can be ablated independently:

  1. Joint-Zoom branch       - a second, cropped-and-zoomed view centered on the joint line,
                                so the network also sees the joint space at higher effective
                                resolution, not just resized-down inside the global view.
  2. Multi-scale fusion      - Stage 3 features (higher spatial resolution, 14x14) folded into
                                Stage 4 (7x7), so finer joint-space detail isn't fully lost to
                                downsampling before the attention/pooling stage.
  3. Joint-Guided CBAM       - nudges the learned spatial attention toward the joint line with
                                a Gaussian prior, encoding "look here first" instead of relying
                                on attention to discover the joint line's location unsupervised.
  4. Asymmetry module        - explicitly computes |medial - lateral| pooled-feature
                                difference, since compartment asymmetry is itself a KL grading
                                cue (e.g. isolated medial-compartment narrowing) that a single
                                symmetric pooled vector cannot represent.

All four are additive and gated by boolean flags in KneeNeXt.__init__, so the ablation
(scripts/train_kneenext.py) can add them one at a time and keep only what helps on validation.

Strict project rule: this file only ever IMPORTS from kl_common.py/measure_jsw.py (read-only -
nothing in those files is modified). Where a piece of existing logic needs changing (the plain
CBAM -> Joint-Guided CBAM; measure_jsw's joint-row finder -> one with a failure/fallback path),
that logic is copied into this file and modified here, never edited in place in the original.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A
from torchvision.models import convnext_tiny

from kl_common import MEAN, STD, IMG_SIZE, apply_clahe, CBAM, coral_probs_to_grade

ZOOM_BAND_FRAC = 0.25      # +-25% of image height around the joint line, per the brief
PRIOR_SIZE = 7             # joint_prior is built directly at the 7x7 feature-map resolution
PRIOR_SIGMA_FRAC = 0.15    # Gaussian band width, as a fraction of image height
JOINT_ROW_FAIL_Z = 1.0     # z-score threshold below which "darkest row" isn't a real dip


# --------------------------------------------------------------------------------------------
# Joint-line detection (copied from measure_jsw.find_joint_row and extended with a failure/
# fallback path - measure_jsw's version always returns *some* row with no notion of
# confidence, which is fine for a validation-only JSW measurement but not safe to feed
# silently into a trainable crop: a bad crop on a feature-less or corrupted image would just
# train the zoom branch on noise. Falls back to the vertical center and reports the failure.)
# --------------------------------------------------------------------------------------------
def detect_joint_row(gray):
    """Returns (row, used_fallback). Darkest horizontal band in the central 50% of the image
    (femur/tibia are bright, the joint space between them is dark), same core idea as
    measure_jsw.find_joint_row, but flags it as a failure (and falls back to the vertical
    center) when that "darkest" row isn't actually a meaningful dip - e.g. a flat, low-
    contrast, or corrupted image - rather than trusting an arbitrary argmin."""
    h, w = gray.shape
    lo, hi = int(h * 0.25), int(h * 0.75)
    row_mean = gray[lo:hi, :].astype(np.float32).mean(axis=1)
    row = int(np.argmin(row_mean)) + lo
    std = row_mean.std()
    z = (row_mean.mean() - row_mean.min()) / (std + 1e-6)
    if z < JOINT_ROW_FAIL_Z:
        return h // 2, True
    return row, False


def gaussian_row_band(center_frac, size=PRIOR_SIZE, sigma_frac=PRIOR_SIGMA_FRAC):
    """A (size, size) float32 array that varies only by row (the joint line runs horizontally
    across the image, so the prior should be a horizontal band, uniform across columns),
    peaking at center_frac (0=top, 1=bottom) with a Gaussian falloff."""
    rows = np.linspace(0.0, 1.0, size, dtype=np.float32)
    band = np.exp(-0.5 * ((rows - center_frac) / sigma_frac) ** 2)
    return np.tile(band[:, None], (1, size)).astype(np.float32)


def crop_zoom_view(img, joint_row, band_frac=ZOOM_BAND_FRAC):
    """Crops a full-width horizontal band of +-band_frac*H around joint_row, then resizes to
    IMG_SIZE x IMG_SIZE - the "zoomed in on the joint line" view."""
    h = img.shape[0]
    half = max(int(band_frac * h), 1)
    top, bot = max(0, joint_row - half), min(h, joint_row + half)
    if bot <= top:   # degenerate edge case (joint_row right at the image border)
        top, bot = 0, h
    crop = img[top:bot, :]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def _to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def _geom_color_aug():
    """Geometric + color augmentation only (no resize/normalize - those happen per-view,
    after the global/zoom split, at different crop sizes). Same transform family/params as
    kl_final's get_transforms(), just reordered to run once on the shared pre-split image so
    both views see an identical augmentation (flip/rotate/brightness), not two independent
    random draws."""
    return A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                      A.RandomBrightnessContrast(0.2, 0.2, p=0.5)])


class KneeNextDataset(Dataset):
    """Returns (global_img, zoom_img, global_prior, zoom_prior, label) for every sample,
    regardless of which KneeNeXt modules are actually switched on - the training loop decides
    which pieces to feed the model. This keeps one dataset implementation valid across the
    whole cumulative ablation instead of a different dataset per configuration.

    Pipeline (order matches kl_final's: CLAHE, then geometric/color augmentation, then
    per-view resize + normalize): load -> CLAHE -> [train only] shared flip/rotate/brightness
    -> detect joint line on this (augmented) CLAHE image -> global view = resize(whole image),
    zoom view = resize(crop band around the joint line). Detecting the joint line AFTER
    augmentation (not before) matters: the crop must match the geometry the network actually
    sees, including any rotation/flip applied.
    """
    def __init__(self, paths, labels, train):
        self.paths, self.labels, self.train = paths, labels, train
        self.aug = _geom_color_aug() if train else None
        self.n_total, self.n_fallback = 0, 0   # joint-detection failure-rate bookkeeping

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_clahe(img)
        if self.aug is not None:
            img = self.aug(image=img)["image"]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        joint_row, fallback = detect_joint_row(gray)
        self.n_total += 1
        self.n_fallback += int(fallback)
        h = gray.shape[0]

        global_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        zoom_img = crop_zoom_view(img, joint_row)
        global_prior = gaussian_row_band(joint_row / h)
        zoom_prior = gaussian_row_band(0.5)   # zoom view is centered on the joint line by construction

        return (_to_tensor(global_img), _to_tensor(zoom_img),
               torch.from_numpy(global_prior), torch.from_numpy(zoom_prior), self.labels[i])

    @property
    def fallback_rate(self):
        return self.n_fallback / self.n_total if self.n_total else 0.0


# --------------------------------------------------------------------------------------------
# Joint-Guided CBAM: a modified copy of kl_common.CBAM (not a subclass, so this file has no
# runtime dependency on CBAM's internals changing underneath it) - channel attention is
# unchanged; spatial attention adds alpha * joint_prior to the learned spatial logit before
# the sigmoid, instead of using the learned logit alone. Submodule names (mlp/avg_pool/
# max_pool/spatial_conv) are kept identical to kl_common.CBAM on purpose, so kl_final's
# trained cbam.* weights still line up and transfer by name (see build_kneenext).
# --------------------------------------------------------------------------------------------
class JointGuidedCBAM(nn.Module):
    """CBAM whose spatial attention is sigmoid(learned_map + alpha * joint_prior) - alpha
    starts at 1.0 (learnable) so the prior has a real say from the first gradient step, but
    training can grow, shrink, or flip it if the data disagrees with "trust the joint line"."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, joint_prior):
        ch_att = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ch_att
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        sp_logit = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        sp_att = torch.sigmoid(sp_logit + self.alpha * joint_prior.unsqueeze(1))
        return x * sp_att, sp_att


class KneeNeXt(nn.Module):
    """ConvNeXt-Tiny backbone (shared weights across the global and zoom views) with four
    optional modules, each gated by a flag so the ablation can add them cumulatively:
    use_zoom, use_multiscale, use_joint_cbam, use_asymmetry. With every flag off, this is
    architecturally equivalent to kl_final (single view, plain CBAM on Stage 4 only) - the
    ablation's "Control" configuration, retrained with KneeNeXt's two-stage schedule so the
    schedule itself isn't a confound when judging the other modules.
    """
    STAGE3_IDX, STAGE4_IDX = 6, 8   # features[:STAGE3_IDX] -> Stage3, features[:STAGE4_IDX] -> Stage4

    def __init__(self, base_convnext, cbam_channels=768, stage3_channels=384,
                use_zoom=False, use_multiscale=False, use_joint_cbam=False, use_asymmetry=False,
                dropout=0.3):
        super().__init__()
        self.use_zoom = use_zoom
        self.use_multiscale = use_multiscale
        self.use_joint_cbam = use_joint_cbam
        self.use_asymmetry = use_asymmetry and use_zoom   # asymmetry needs the global view only, but is
                                                           # meaningless as an ablation step without zoom too

        self.features = base_convnext.features
        if use_multiscale:
            self.stage3_proj = nn.Conv2d(stage3_channels, cbam_channels, kernel_size=1)
            self.stage3_down = nn.AvgPool2d(kernel_size=2)
        self.cbam = JointGuidedCBAM(cbam_channels) if use_joint_cbam else CBAM(cbam_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        n_groups = 1 + int(use_zoom) + int(self.use_asymmetry)
        in_dim = cbam_channels * n_groups
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_dim, 4)

        self.last_attn = None    # most recent global-view spatial attention map (Grad-CAM/vis parity)

    def _extract(self, x):
        h = x
        for i in range(self.STAGE3_IDX):
            h = self.features[i](h)
        stage3 = h
        for i in range(self.STAGE3_IDX, self.STAGE4_IDX):
            h = self.features[i](h)
        return stage3, h   # (384x14x14, 768x7x7)

    def _fuse(self, stage3, stage4):
        if not self.use_multiscale:
            return stage4
        return stage4 + self.stage3_down(self.stage3_proj(stage3))

    def _process_view(self, x, joint_prior):
        stage3, stage4 = self._extract(x)
        fused = self._fuse(stage3, stage4)
        if self.use_joint_cbam:
            attended, attn = self.cbam(fused, joint_prior)
        else:
            attended, attn = self.cbam(fused)
        pooled = self.pool(attended).flatten(1)
        return attended, pooled, attn

    def forward(self, global_x, zoom_x=None, global_prior=None, zoom_prior=None):
        global_attended, global_pooled, global_attn = self._process_view(global_x, global_prior)
        self.last_attn = global_attn.detach()
        feats = [global_pooled]

        if self.use_zoom:
            _, zoom_pooled, _ = self._process_view(zoom_x, zoom_prior)
            feats.append(zoom_pooled)
            if self.use_asymmetry:
                # Drop the central column (tibial spines / intercondylar notch) before
                # splitting medial/lateral, same reasoning as measure_jsw.py's compartment
                # split - that region isn't reliably assignable to either side.
                left = global_attended[:, :, :, :3]
                right = global_attended[:, :, :, 4:]
                l = self.pool(left).flatten(1)
                r = self.pool(right).flatten(1)
                feats.append((l - r).abs())

        x = torch.cat(feats, dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


CONFIGS = ["control", "joint_zoom", "multiscale", "joint_cbam", "asymmetry"]


def config_flags(config):
    """Cumulative ablation flags for a named config - each config turns on everything the
    previous ones did, plus one more module."""
    order = ["joint_zoom", "multiscale", "joint_cbam", "asymmetry"]
    on = set(order[:order.index(config) + 1]) if config != "control" else set()
    return {"use_zoom": "joint_zoom" in on, "use_multiscale": "multiscale" in on,
           "use_joint_cbam": "joint_cbam" in on, "use_asymmetry": "asymmetry" in on}


def build_kneenext(config, ckpt_dir, device, init_from="kl_final_best.pt"):
    """Builds a KneeNeXt for the given ablation config, initializing features.*/cbam.* from
    kl_final's checkpoint (read-only) by copying any state_dict keys with matching names and
    shapes - same pattern as kl_common.build_hybrid1/build_hybrid2. JointGuidedCBAM keeps
    CBAM's submodule names (mlp/avg_pool/max_pool/spatial_conv) so its weights line up too;
    only its extra `alpha` parameter has no counterpart and stays at its own init (1.0)."""
    base = convnext_tiny(weights=None)
    flags = config_flags(config)
    model = KneeNeXt(base, **flags)
    if init_from:
        src_state = torch.load(os.path.join(ckpt_dir, init_from), map_location=device)
        model_state = model.state_dict()
        n_loaded = 0
        for k, v in src_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                n_loaded += 1
        model.load_state_dict(model_state)
        print(f"KneeNeXt[{config}]: initialized {n_loaded} tensors from {init_from} (features.*/cbam.*)")
    return model.to(device)


def prepare_views(rgb, already_clahe=False):
    """Given an already-loaded RGB image (any size), returns (global_img, zoom_img,
    global_tensor, zoom_tensor, global_prior, zoom_prior, fallback) - deterministic (no
    augmentation), matching KneeNextDataset's eval-time (train=False) pipeline exactly. Used
    by predict_kneenext.py and evaluate_kneenext.py so both build views from an arbitrary
    image (not just a dataset path) the same way. already_clahe=True skips re-applying CLAHE
    (e.g. predict_kneenext.py's auto-crop path may already have it from an earlier step)."""
    img = rgb if already_clahe else apply_clahe(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    joint_row, fallback = detect_joint_row(gray)
    h = gray.shape[0]
    global_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    zoom_img = crop_zoom_view(img, joint_row)
    global_prior = gaussian_row_band(joint_row / h)
    zoom_prior = gaussian_row_band(0.5)
    return (global_img, zoom_img, _to_tensor(global_img), _to_tensor(zoom_img),
           global_prior, zoom_prior, fallback)


def run_eval_kneenext(model, loader, device, use_amp=True):
    """Returns (predicted grades, true grades, 4-dim CORAL sigmoid probs) for a whole
    KneeNextDataset loader. Mirrors kl_common.run_eval_probs, adapted for KneeNeXt's dual-
    view forward signature; zoom/prior tensors are always loaded (KneeNextDataset always
    returns them) but only actually passed to the model when its flags call for them."""
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for g, z, gp, zp, y in loader:
            g, z, gp, zp = g.to(device), z.to(device), gp.to(device), zp.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(g, z if model.use_zoom else None,
                               gp if model.use_joint_cbam else None, zp if model.use_joint_cbam else None)
            p = torch.sigmoid(logits.float())
            preds.append(coral_probs_to_grade(p).cpu())
            labels.append(y)
            probs.append(p.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy()


def grad_cam_view(model, global_x, zoom_x, global_prior, zoom_prior, feature_module):
    """Grad-CAM for KneeNeXt's global view's attended feature map (same target convention as
    kl_common.grad_cam: sum of the 4 CORAL threshold logits, since there's no single softmax
    class score). feature_module should be model.features[model.STAGE3_IDX - 1] or similar -
    passed in explicitly rather than hardcoded, since which stage to hook depends on whether
    multiscale fusion is on."""
    activations, gradients = {}, {}
    def fwd(_, __, out): activations["v"] = out
    def bwd(_, grad_in, grad_out): gradients["v"] = grad_out[0]
    h1 = feature_module.register_forward_hook(fwd)
    h2 = feature_module.register_full_backward_hook(bwd)
    model.zero_grad()
    logits = model(global_x, zoom_x, global_prior, zoom_prior)
    logits.sum().backward()
    h1.remove(); h2.remove()
    acts, grads = activations["v"], gradients["v"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    import torch.nn.functional as F
    cam = F.relu((weights * acts).sum(dim=1)).squeeze().detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, torch.sigmoid(logits.detach()).squeeze().cpu().numpy()
