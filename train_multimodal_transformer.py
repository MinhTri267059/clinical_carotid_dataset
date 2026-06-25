#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multimodal_transformer.py
================================
Multimodal Transformer với Khởi tạo Phổ quát (Self-Supervised Pre-training)
cho bài toán phân loại xơ vữa động mạch cảnh.

Kiến trúc:
  - Nhánh Ảnh  : ViT-style Patch Embedding (16x16 patches trên ảnh 128x128)
  - Nhánh Bảng : FT-Transformer style Feature Tokenizer
  - Fusion      : Multi-Head Self-Attention Transformer chung với [CLS] token
  - Phase 1     : Tiền huấn luyện Masked Multimodal Reconstruction (SSP)
  - Phase 2     : Tinh chỉnh phân loại nhị phân có giám sát

Outputs:
  outputs/training.log                      - Log đầy đủ
  outputs/multimodal_transformer_pretrained.pth - Trọng số sau pre-training
  outputs/multimodal_transformer_best.pth   - Trọng số tốt nhất (theo val F1)
  outputs/training_curves.png               - Biểu đồ Loss/Acc/F1
  outputs/confusion_matrix.png              - Ma trận nhầm lẫn trên tập test
"""

import os
import math
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ======================================================================
# 0. THIẾT LẬP & CẤU HÌNH
# ======================================================================

WORKSPACE   = Path("/Users/minhtri86/Downloads/clinical_carotid_dataset_v3")
CSV_PATH    = WORKSPACE / "carotid_clinical_dataset_300cases.csv"
IMAGES_DIR  = WORKSPACE / "CAROTID_IMAGES"
OUTPUT_DIR  = WORKSPACE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE           = OUTPUT_DIR / "training.log"
MODEL_PRETRAIN_PATH = OUTPUT_DIR / "multimodal_transformer_pretrained.pth"
MODEL_BEST_PATH    = OUTPUT_DIR / "multimodal_transformer_best.pth"
CURVES_PATH        = OUTPUT_DIR / "training_curves.png"
CM_PATH            = OUTPUT_DIR / "confusion_matrix.png"

# Hyperparameters
IMG_SIZE            = 128      # resize ảnh về 128x128
PATCH_SIZE          = 16       # kích thước mỗi patch
N_PATCHES_PER_IMG   = (IMG_SIZE // PATCH_SIZE) ** 2   # 64
MAX_IMAGES          = 5        # số ảnh tối đa mỗi bệnh nhân (padding nếu thiếu)
N_IMG_TOKENS        = N_PATCHES_PER_IMG * MAX_IMAGES   # 320

EMBED_DIM   = 128
N_HEADS     = 8
N_LAYERS    = 4
FF_DIM      = 256
DROPOUT     = 0.1

BATCH_SIZE      = 16
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 60
LR_PRETRAIN     = 1e-4
LR_FINETUNE     = 5e-5
MASK_RATIO      = 0.15
TEST_SIZE       = 0.20
RANDOM_SEED     = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đặc trưng đầu vào
CONTINUOUS_COLS = [
    'Age', 'Lp(a)_mg_dL', 'ApoB_mg_dL', 'LDL_C_mg_dL',
    'Triglyceride_mg_dL', 'Total_Cholesterol_mg_dL',
    'Non_HDL_mg_dL', 'IMT_mm', 'Baseline_Risk_Score'
]
CATEGORICAL_COLS = ['Sex']   # Male→0, Female→1
TARGET_COL       = 'Plaque_present'
N_CONT           = len(CONTINUOUS_COLS)
N_CAT            = len(CATEGORICAL_COLS)
N_TAB_TOKENS     = N_CONT + N_CAT

# ======================================================================
# 1. LOGGER
# ======================================================================

def setup_logger():
    logger = logging.getLogger('MultimodalTransformer')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# ======================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ======================================================================

def load_and_preprocess():
    logger.info("=== TIỀN XỬ LÝ DỮ LIỆU ===")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Tổng số bệnh nhân: {len(df)}")

    # Encode giới tính: Male=0, Female=1
    df['Sex'] = (df['Sex'] == 'Female').astype(int)

    X_cont = df[CONTINUOUS_COLS].values.astype(np.float32)
    X_cat  = df[CATEGORICAL_COLS].values.astype(np.int64)
    y      = df[TARGET_COL].values.astype(np.int64)
    img_lists = df['Associated_Images'].tolist()

    # Stratified split 80/20
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    u_tr, c_tr = np.unique(y[train_idx], return_counts=True)
    u_te, c_te = np.unique(y[test_idx],  return_counts=True)
    logger.info(f"Phân phối Train: {dict(zip(u_tr.tolist(), c_tr.tolist()))}")
    logger.info(f"Phân phối Test : {dict(zip(u_te.tolist(), c_te.tolist()))}")

    # Chuẩn hóa trên train, áp lên test
    scaler = StandardScaler()
    X_norm = X_cont.copy()
    X_norm[train_idx] = scaler.fit_transform(X_cont[train_idx])
    X_norm[test_idx]  = scaler.transform(X_cont[test_idx])

    return dict(
        X_cont=X_norm, X_cat=X_cat, y=y,
        img_lists=img_lists, train_idx=train_idx, test_idx=test_idx
    )

# ======================================================================
# 3. DATASET
# ======================================================================

def _load_single_image(fname):
    """Đọc, resize và chuẩn hóa một ảnh. Trả về tensor (1, H, W)."""
    img  = Image.open(IMAGES_DIR / fname).convert('L')
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)   # (1, H, W)


def _get_patient_images(img_str):
    """
    Tải ảnh bệnh nhân, padding đến MAX_IMAGES ảnh đen.
    Trả về:
      images   : (MAX_IMAGES, 1, H, W)
      pad_mask : (MAX_IMAGES,) bool — True = ảnh padding (bỏ qua)
    """
    names = [s.strip() for s in img_str.split(',') if s.strip()]
    tensors, is_pad = [], []
    for name in names[:MAX_IMAGES]:
        tensors.append(_load_single_image(name))
        is_pad.append(False)
    while len(tensors) < MAX_IMAGES:
        tensors.append(torch.zeros(1, IMG_SIZE, IMG_SIZE))
        is_pad.append(True)
    return torch.stack(tensors), torch.tensor(is_pad, dtype=torch.bool)


class CarotidDataset(Dataset):
    def __init__(self, data, indices):
        self.X_cont    = data['X_cont'][indices]
        self.X_cat     = data['X_cat'][indices]
        self.y         = data['y'][indices]
        self.img_lists = [data['img_lists'][i] for i in indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_cont  = torch.tensor(self.X_cont[idx], dtype=torch.float32)
        x_cat   = torch.tensor(self.X_cat[idx],  dtype=torch.long)
        label   = torch.tensor(self.y[idx],       dtype=torch.long)
        images, pad_mask = _get_patient_images(self.img_lists[idx])
        return x_cont, x_cat, images, pad_mask, label

# ======================================================================
# 4. KIẾN TRÚC MÔ HÌNH
# ======================================================================

class PatchEmbedding(nn.Module):
    """ViT-style: chia ảnh thành patches rồi chiếu lên embed_dim."""

    def __init__(self, img_size=128, patch_size=16, in_ch=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        patch_flat_dim  = in_ch * patch_size * patch_size
        self.proj = nn.Linear(patch_flat_dim, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)       # (B,C,nH,nW,p,p)
        x = x.contiguous().view(B, C, -1, p, p)      # (B,C,n,p,p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()    # (B,n,C,p,p)
        x = x.view(B, self.n_patches, -1)             # (B,n,C*p*p)
        return self.proj(x) + self.pos_embed          # (B,n,d)


class TabularTokenizer(nn.Module):
    """
    FT-Transformer style:
      - Mỗi đặc trưng liên tục: Linear(1 → d) riêng biệt
      - Mỗi đặc trưng phân loại: Embedding(vocab → d)
    """

    def __init__(self, n_cont, cat_vocab_sizes, embed_dim=128):
        super().__init__()
        n_cat = len(cat_vocab_sizes)
        # Continuous: learnable weight + bias cho từng đặc trưng
        self.W = nn.Parameter(torch.randn(n_cont, embed_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_cont, embed_dim))
        # Categorical embeddings
        self.cat_embs = nn.ModuleList([
            nn.Embedding(v, embed_dim) for v in cat_vocab_sizes
        ])
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_cont + n_cat, embed_dim) * 0.02
        )

    def forward(self, x_cont, x_cat):
        # Continuous tokens: (B, n_cont, d)
        cont_tok = x_cont.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        # Categorical tokens: (B, n_cat, d)
        cat_toks = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1
        )
        tokens = torch.cat([cont_tok, cat_toks], dim=1)  # (B, n_tab, d)
        return tokens + self.pos_embed


class MultimodalTransformer(nn.Module):
    """
    Unified Multimodal Transformer:
      [CLS] | tabular_tokens | image_patch_tokens → Transformer → CLS → classifier
    """

    def __init__(self, img_size=128, patch_size=16,
                 n_cont=9, cat_vocab_sizes=(2,),
                 embed_dim=128, n_heads=8, n_layers=4,
                 ff_dim=256, dropout=0.1, max_images=5):
        super().__init__()
        self.embed_dim        = embed_dim
        self.n_tab_tokens     = n_cont + len(cat_vocab_sizes)
        self.n_patches_per_img = (img_size // patch_size) ** 2
        self.max_images       = max_images
        self.n_img_tokens     = self.n_patches_per_img * max_images

        # Sub-modules
        self.patch_embed  = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.tab_tokenizer = TabularTokenizer(n_cont, cat_vocab_sizes, embed_dim)

        # Đặc biệt
        self.cls_token    = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.seg_embed    = nn.Embedding(3, embed_dim)  # 0=cls,1=tab,2=img

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (fine-tuning)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)
        )

        # Reconstruction head (pre-training — MSE trong embedding space)
        # Không cần chiếu ra raw feature vì ta học reconstruction trên embedding

    # ------------------------------------------------------------------
    def _tokenize(self, x_cont, x_cat, images, img_pad_mask):
        """
        Hợp nhất tất cả token + tạo key_padding_mask.
        Trả về: tokens (B,L,d), mask (B,L) True=bỏ qua
        """
        B = x_cont.size(0)

        # Tabular tokens
        tab_tok = self.tab_tokenizer(x_cont, x_cat)   # (B, n_tab, d)

        # Image tokens từng ảnh
        img_toks = []
        for i in range(self.max_images):
            img_toks.append(self.patch_embed(images[:, i]))  # (B, 64, d)
        img_tok = torch.cat(img_toks, dim=1)                 # (B, 320, d)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d)

        # Segment embeddings
        seg_cls = self.seg_embed(torch.full((B, 1), 0, dtype=torch.long, device=x_cont.device))
        seg_tab = self.seg_embed(torch.full((B, self.n_tab_tokens), 1, dtype=torch.long, device=x_cont.device))
        seg_img = self.seg_embed(torch.full((B, self.n_img_tokens), 2, dtype=torch.long, device=x_cont.device))

        # Nối: [CLS | tab | img]
        tokens = torch.cat([
            cls + seg_cls,
            tab_tok + seg_tab,
            img_tok + seg_img
        ], dim=1)                                            # (B, 1+n_tab+n_img, d)

        # Key padding mask: True = bỏ qua
        cls_m = torch.zeros(B, 1, dtype=torch.bool, device=x_cont.device)
        tab_m = torch.zeros(B, self.n_tab_tokens, dtype=torch.bool, device=x_cont.device)
        # Mở rộng img_pad_mask: (B, MAX_IMAGES) → (B, n_img_tokens)
        img_m = img_pad_mask.unsqueeze(-1)\
                    .expand(B, self.max_images, self.n_patches_per_img)\
                    .reshape(B, self.n_img_tokens)

        kp_mask = torch.cat([cls_m, tab_m, img_m], dim=1)  # (B, L)
        return tokens, kp_mask

    # ------------------------------------------------------------------
    def forward(self, x_cont, x_cat, images, img_pad_mask, mask_bool=None):
        """
        mask_bool: (B, L) bool, True = token bị che (chỉ dùng trong pre-training)
        - Nếu mask_bool is None → fine-tuning → trả về logits
        - Nếu mask_bool is not None → pre-training → trả về (encoded, original_tokens, kp_mask)
        """
        tokens, kp_mask = self._tokenize(x_cont, x_cat, images, img_pad_mask)

        if mask_bool is not None:
            original = tokens.clone()
            tokens = tokens.masked_fill(mask_bool.unsqueeze(-1), 0.0)

        encoded = self.transformer(tokens, src_key_padding_mask=kp_mask)
        encoded = self.norm(encoded)

        if mask_bool is not None:
            return encoded, original, kp_mask

        cls_out = encoded[:, 0]               # (B, d)
        return self.classifier(cls_out)       # (B, 2)

# ======================================================================
# 5. PRE-TRAINING — MASKED MULTIMODAL RECONSTRUCTION
# ======================================================================

def _make_mask(tokens, kp_mask, mask_ratio):
    """Tạo boolean mask ngẫu nhiên trên các token hợp lệ (bỏ CLS)."""
    B, L, _ = tokens.shape
    valid       = ~kp_mask                           # True = hợp lệ
    valid[:, 0] = False                              # không mask CLS
    rand        = torch.rand(B, L, device=tokens.device)
    rand[~valid] = 2.0
    return rand < mask_ratio                         # (B, L)


def pretrain_step(model, batch, optimizer):
    x_cont, x_cat, images, pad_mask, _ = [b.to(DEVICE) for b in batch]

    with torch.no_grad():
        tokens, kp_mask = model._tokenize(x_cont, x_cat, images, pad_mask)

    mask = _make_mask(tokens, kp_mask, MASK_RATIO)
    encoded, original, _ = model(x_cont, x_cat, images, pad_mask, mask_bool=mask)

    # Reconstruction loss (MSE trong không gian embedding)
    loss = F.mse_loss(encoded[mask], original[mask].detach())

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# ======================================================================
# 6. FINE-TUNING — SUPERVISED CLASSIFICATION
# ======================================================================

def finetune_step(model, batch, optimizer, criterion):
    x_cont, x_cat, images, pad_mask, labels = [b.to(DEVICE) for b in batch]
    logits = model(x_cont, x_cat, images, pad_mask)
    loss   = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), logits.detach().cpu(), labels.detach().cpu()


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        x_cont, x_cat, images, pad_mask, labels = [b.to(DEVICE) for b in batch]
        logits  = model(x_cont, x_cat, images, pad_mask)
        tot_loss += criterion(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits  = torch.cat(all_logits)
    labels  = torch.cat(all_labels).numpy()
    probs   = F.softmax(logits, dim=1)[:, 1].numpy()
    preds   = logits.argmax(1).numpy()
    n       = len(labels)

    avg_loss = tot_loss / n
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    roc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return avg_loss, acc, prec, rec, f1, roc, preds, labels, probs

# ======================================================================
# 7. VẼ BIỂU ĐỒ
# ======================================================================

def plot_curves(pre_losses, tr_losses, va_losses,
                tr_accs, va_accs, tr_f1s, va_f1s):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Kết Quả Huấn Luyện — Multimodal Transformer',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(pre_losses, color='#7B2D8B', lw=1.5)
    ax.set_title('Loss — Tiền Huấn Luyện (SSP)'); ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(tr_losses, label='Train', color='#2196F3', lw=1.5)
    ax.plot(va_losses, label='Validation', color='#FF5722', lw=1.5)
    ax.set_title('Loss — Tinh Chỉnh'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tr_accs, label='Train', color='#2196F3', lw=1.5)
    ax.plot(va_accs, label='Validation', color='#FF5722', lw=1.5)
    ax.set_title('Accuracy — Tinh Chỉnh'); ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(tr_f1s, label='Train', color='#2196F3', lw=1.5)
    ax.plot(va_f1s, label='Validation', color='#FF5722', lw=1.5)
    ax.set_title('F1-Score — Tinh Chỉnh'); ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ: {CURVES_PATH}")


def plot_cm(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    classes = ['Không Có Mảng (0)', 'Có Mảng Xơ Vữa (1)']
    ticks   = np.arange(2)
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=9)
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=15, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Nhãn Dự Đoán', fontsize=11)
    ax.set_ylabel('Nhãn Thực Tế', fontsize=11)
    ax.set_title('Ma Trận Nhầm Lẫn — Tập Test', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu confusion matrix: {CM_PATH}")

# ======================================================================
# 8. MAIN
# ======================================================================

def main():
    bar = "=" * 62
    logger.info(bar)
    logger.info("  MULTIMODAL TRANSFORMER — CAROTID ATHEROSCLEROSIS")
    logger.info("  Khởi tạo Phổ quát (Self-Supervised Pre-training)")
    logger.info(bar)
    logger.info(f"Device : {DEVICE}")
    logger.info(f"Config : embed_dim={EMBED_DIM}, heads={N_HEADS}, "
                f"layers={N_LAYERS}, ff_dim={FF_DIM}")
    logger.info(f"Image  : size={IMG_SIZE}, patch={PATCH_SIZE}, "
                f"patches/img={N_PATCHES_PER_IMG}, max_imgs={MAX_IMAGES}")
    logger.info(f"Tokens : tab={N_TAB_TOKENS}, img={N_IMG_TOKENS}, "
                f"total={1+N_TAB_TOKENS+N_IMG_TOKENS} (+CLS)")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Dữ liệu ──────────────────────────────────────────────────────
    data = load_and_preprocess()
    train_ds = CarotidDataset(data, data['train_idx'])
    test_ds  = CarotidDataset(data, data['test_idx'])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

    # ── Mô hình ───────────────────────────────────────────────────────
    model = MultimodalTransformer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        n_cont=N_CONT, cat_vocab_sizes=(2,),
        embed_dim=EMBED_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM,
        dropout=DROPOUT, max_images=MAX_IMAGES
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Số tham số huấn luyện: {n_params:,}")

    # ==================================================================
    # GIAI ĐOẠN 1 — TIỀN HUẤN LUYỆN TỰ GIÁM SÁT
    # ==================================================================
    logger.info("\n" + bar)
    logger.info(" GIAI ĐOẠN 1: TIỀN HUẤN LUYỆN TỰ GIÁM SÁT (SSP)")
    logger.info(f" Mask ratio={MASK_RATIO*100:.0f}% | Epochs={PRETRAIN_EPOCHS} | LR={LR_PRETRAIN}")
    logger.info(bar)

    opt_pre  = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    sched_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, T_max=PRETRAIN_EPOCHS)
    pre_losses = []
    t0 = time.time()

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        ep_losses = [pretrain_step(model, batch, opt_pre) for batch in train_dl]
        sched_pre.step()
        avg = np.mean(ep_losses)
        pre_losses.append(avg)
        logger.info(f"[Pre] Epoch {ep:3d}/{PRETRAIN_EPOCHS} | "
                    f"MSE Loss={avg:.6f} | "
                    f"LR={sched_pre.get_last_lr()[0]:.2e} | "
                    f"T={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PRETRAIN_PATH)
    logger.info(f"\n→ Lưu trọng số tiền huấn luyện: {MODEL_PRETRAIN_PATH}")

    # ==================================================================
    # GIAI ĐOẠN 2 — TINH CHỈNH CÓ GIÁM SÁT
    # ==================================================================
    logger.info("\n" + bar)
    logger.info(" GIAI ĐOẠN 2: TINH CHỈNH CÓ GIÁM SÁT (SUPERVISED FINE-TUNING)")
    logger.info(f" Epochs={FINETUNE_EPOCHS} | LR={LR_FINETUNE}")
    logger.info(bar)

    criterion = nn.CrossEntropyLoss()
    opt_ft    = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    sched_ft  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=FINETUNE_EPOCHS)

    tr_losses, va_losses = [], []
    tr_accs,   va_accs   = [], []
    tr_f1s,    va_f1s    = [], []
    best_f1 = -1.0
    t0 = time.time()

    for ep in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        ep_losses, ep_logits, ep_labels = [], [], []
        for batch in train_dl:
            loss, logits, labels = finetune_step(model, batch, opt_ft, criterion)
            ep_losses.append(loss)
            ep_logits.append(logits); ep_labels.append(labels)
        sched_ft.step()

        # Train metrics
        all_log = torch.cat(ep_logits)
        all_lbl = torch.cat(ep_labels).numpy()
        all_prd = all_log.argmax(1).numpy()
        tr_loss = np.mean(ep_losses)
        tr_acc  = accuracy_score(all_lbl, all_prd)
        tr_f1   = f1_score(all_lbl, all_prd, zero_division=0)

        # Val metrics
        va_loss, va_acc, va_prec, va_rec, va_f1, va_roc, *_ = evaluate(
            model, test_dl, criterion
        )
        model.train()

        tr_losses.append(tr_loss); va_losses.append(va_loss)
        tr_accs.append(tr_acc);    va_accs.append(va_acc)
        tr_f1s.append(tr_f1);      va_f1s.append(va_f1)

        flag = ''
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_f1': va_f1, 'val_roc': va_roc, 'val_acc': va_acc
            }, MODEL_BEST_PATH)
            flag = '  ← BEST ✓'

        logger.info(
            f"[FT] Ep {ep:3d}/{FINETUNE_EPOCHS} | "
            f"Tr Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"Va Loss={va_loss:.4f} Acc={va_acc:.4f} F1={va_f1:.4f} "
            f"AUC={va_roc:.4f} | T={time.time()-t0:.0f}s{flag}"
        )

    # ==================================================================
    # ĐÁNH GIÁ CUỐI CÙNG
    # ==================================================================
    logger.info("\n" + bar)
    logger.info(" ĐÁNH GIÁ CUỐI CÙNG — TẬP TEST (trọng số tốt nhất)")
    logger.info(bar)

    ckpt = torch.load(MODEL_BEST_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Tải trọng số từ epoch {ckpt['epoch']}")

    te_loss, te_acc, te_prec, te_rec, te_f1, te_roc, \
        te_preds, te_labels, _ = evaluate(model, test_dl, criterion)

    logger.info(f"\n{'─'*40}")
    logger.info(f"  Accuracy  : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    logger.info(f"  Precision : {te_prec:.4f}")
    logger.info(f"  Recall    : {te_rec:.4f}")
    logger.info(f"  F1-Score  : {te_f1:.4f}")
    logger.info(f"  ROC-AUC   : {te_roc:.4f}")
    logger.info(f"  Loss      : {te_loss:.4f}")
    logger.info(f"{'─'*40}\n")

    # ==================================================================
    # XUẤT BIỂU ĐỒ
    # ==================================================================
    plot_curves(pre_losses, tr_losses, va_losses,
                tr_accs, va_accs, tr_f1s, va_f1s)
    plot_cm(te_labels, te_preds)

    logger.info("\n" + bar)
    logger.info(" HOÀN THÀNH!")
    logger.info(f"  Mô hình tốt nhất : {MODEL_BEST_PATH}")
    logger.info(f"  Log huấn luyện   : {LOG_FILE}")
    logger.info(f"  Biểu đồ curves   : {CURVES_PATH}")
    logger.info(f"  Confusion Matrix : {CM_PATH}")
    logger.info(bar)


if __name__ == '__main__':
    main()
