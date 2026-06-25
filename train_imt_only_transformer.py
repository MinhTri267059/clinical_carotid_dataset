#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_imt_only_transformer.py
================================
Biến thể Ablation Study: chỉ sử dụng DUY NHẤT 1 ảnh IMT
cho CẢ HAI nhãn (Class 0 và Class 1).

Mục đích: Kiểm tra xem mô hình gốc đạt 100% có thực sự
học từ nội dung lâm sàng + ảnh IMT, hay chỉ "đếm số ảnh"
(Class 1 có 5 ảnh, Class 0 chỉ có 1 ảnh).

So sánh với: train_multimodal_transformer.py (dùng tối đa 5 ảnh)

Outputs → clinical_carotid_dataset_v3/outputs_imt_only/
"""

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
# 0. CẤU HÌNH — CHỈ THAY ĐỔI MAX_IMAGES = 1
# ======================================================================

WORKSPACE    = Path("/Users/minhtri86/Downloads/clinical_carotid_dataset_v3")
CSV_PATH     = WORKSPACE / "carotid_clinical_dataset_300cases.csv"
IMAGES_DIR   = WORKSPACE / "CAROTID_IMAGES"
OUTPUT_DIR   = WORKSPACE / "outputs_imt_only"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE            = OUTPUT_DIR / "training.log"
MODEL_PRETRAIN_PATH = OUTPUT_DIR / "imt_only_pretrained.pth"
MODEL_BEST_PATH     = OUTPUT_DIR / "imt_only_best.pth"
CURVES_PATH         = OUTPUT_DIR / "training_curves.png"
CM_PATH             = OUTPUT_DIR / "confusion_matrix.png"
COMPARE_PATH        = OUTPUT_DIR / "comparison_original_vs_imt_only.png"

# ★ KEY CHANGE: chỉ dùng 1 ảnh IMT
IMG_SIZE          = 128
PATCH_SIZE        = 16
N_PATCHES_PER_IMG = (IMG_SIZE // PATCH_SIZE) ** 2   # 64
MAX_IMAGES        = 1                                 # ← chỉ 1 ảnh IMT
N_IMG_TOKENS      = N_PATCHES_PER_IMG * MAX_IMAGES   # 64 (thay vì 320)

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

CONTINUOUS_COLS = [
    'Age', 'Lp(a)_mg_dL', 'ApoB_mg_dL', 'LDL_C_mg_dL',
    'Triglyceride_mg_dL', 'Total_Cholesterol_mg_dL',
    'Non_HDL_mg_dL', 'IMT_mm', 'Baseline_Risk_Score'
]
CATEGORICAL_COLS = ['Sex']
TARGET_COL       = 'Plaque_present'
N_CONT           = len(CONTINUOUS_COLS)
N_CAT            = len(CATEGORICAL_COLS)
N_TAB_TOKENS     = N_CONT + N_CAT

# ======================================================================
# 1. LOGGER
# ======================================================================

def setup_logger():
    logger = logging.getLogger('IMT_Only_Transformer')
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
    logger.info("=== TIỀN XỬ LÝ DỮ LIỆU (CHỈ DÙNG ẢNH IMT) ===")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Tổng số bệnh nhân: {len(df)}")

    df['Sex'] = (df['Sex'] == 'Female').astype(int)

    X_cont    = df[CONTINUOUS_COLS].values.astype(np.float32)
    X_cat     = df[CATEGORICAL_COLS].values.astype(np.int64)
    y         = df[TARGET_COL].values.astype(np.int64)

    # ★ Chỉ lấy ảnh IMT đầu tiên (luôn là *_IMT.png)
    # Dù bệnh nhân có 1 hay 5 ảnh, chỉ dùng ảnh đầu tiên
    imt_images = []
    for img_str in df['Associated_Images']:
        first_img = img_str.split(',')[0].strip()
        imt_images.append(first_img)
    logger.info("★ Chỉ sử dụng ảnh IMT đầu tiên cho tất cả bệnh nhân")
    logger.info(f"  Ví dụ Class 0: {imt_images[0]}")
    logger.info(f"  Ví dụ Class 1: {[imt_images[i] for i,v in enumerate(y) if v==1][0]}")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    u_tr, c_tr = np.unique(y[train_idx], return_counts=True)
    u_te, c_te = np.unique(y[test_idx],  return_counts=True)
    logger.info(f"Phân phối Train: {dict(zip(u_tr.tolist(), c_tr.tolist()))}")
    logger.info(f"Phân phối Test : {dict(zip(u_te.tolist(), c_te.tolist()))}")

    scaler = StandardScaler()
    X_norm = X_cont.copy()
    X_norm[train_idx] = scaler.fit_transform(X_cont[train_idx])
    X_norm[test_idx]  = scaler.transform(X_cont[test_idx])

    return dict(
        X_cont=X_norm, X_cat=X_cat, y=y,
        imt_images=imt_images,
        train_idx=train_idx, test_idx=test_idx
    )

# ======================================================================
# 3. DATASET — CHỈ 1 ẢNH IMT
# ======================================================================

def _load_image(fname):
    img = Image.open(IMAGES_DIR / fname).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)   # (1, H, W)


class IMTOnlyDataset(Dataset):
    """Chỉ dùng ảnh IMT đầu tiên, bất kể nhãn bệnh nhân."""

    def __init__(self, data, indices):
        self.X_cont     = data['X_cont'][indices]
        self.X_cat      = data['X_cat'][indices]
        self.y          = data['y'][indices]
        self.imt_images = [data['imt_images'][i] for i in indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_cont = torch.tensor(self.X_cont[idx], dtype=torch.float32)
        x_cat  = torch.tensor(self.X_cat[idx],  dtype=torch.long)
        label  = torch.tensor(self.y[idx],       dtype=torch.long)

        # Chỉ 1 ảnh IMT → shape (1, 1, H, W) để khớp với model
        img     = _load_image(self.imt_images[idx])   # (1, H, W)
        images  = img.unsqueeze(0)                     # (1, 1, H, W) = (MAX_IMAGES, C, H, W)
        pad_mask = torch.tensor([False], dtype=torch.bool)  # (1,) không có padding
        return x_cont, x_cat, images, pad_mask, label

# ======================================================================
# 4. KIẾN TRÚC MÔ HÌNH (giữ nguyên, chỉ MAX_IMAGES=1)
# ======================================================================

class PatchEmbedding(nn.Module):
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
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.n_patches, -1)
        return self.proj(x) + self.pos_embed


class TabularTokenizer(nn.Module):
    def __init__(self, n_cont, cat_vocab_sizes, embed_dim=128):
        super().__init__()
        n_cat   = len(cat_vocab_sizes)
        self.W  = nn.Parameter(torch.randn(n_cont, embed_dim) * 0.02)
        self.b  = nn.Parameter(torch.zeros(n_cont, embed_dim))
        self.cat_embs = nn.ModuleList([
            nn.Embedding(v, embed_dim) for v in cat_vocab_sizes
        ])
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_cont + n_cat, embed_dim) * 0.02
        )

    def forward(self, x_cont, x_cat):
        cont_tok = x_cont.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        cat_toks = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1
        )
        tokens = torch.cat([cont_tok, cat_toks], dim=1)
        return tokens + self.pos_embed


class MultimodalTransformerIMTOnly(nn.Module):
    """Kiến trúc giống hệt mô hình gốc nhưng MAX_IMAGES=1."""

    def __init__(self, img_size=128, patch_size=16,
                 n_cont=9, cat_vocab_sizes=(2,),
                 embed_dim=128, n_heads=8, n_layers=4,
                 ff_dim=256, dropout=0.1, max_images=1):
        super().__init__()
        self.embed_dim         = embed_dim
        self.n_tab_tokens      = n_cont + len(cat_vocab_sizes)
        self.n_patches_per_img = (img_size // patch_size) ** 2
        self.max_images        = max_images
        self.n_img_tokens      = self.n_patches_per_img * max_images

        self.patch_embed   = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.tab_tokenizer = TabularTokenizer(n_cont, cat_vocab_sizes, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.seg_embed = nn.Embedding(3, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)
        )

    def _tokenize(self, x_cont, x_cat, images, img_pad_mask):
        B = x_cont.size(0)
        tab_tok = self.tab_tokenizer(x_cont, x_cat)

        img_toks = []
        for i in range(self.max_images):
            img_toks.append(self.patch_embed(images[:, i]))
        img_tok = torch.cat(img_toks, dim=1)

        cls = self.cls_token.expand(B, -1, -1)

        seg_cls = self.seg_embed(torch.full((B, 1), 0, dtype=torch.long, device=x_cont.device))
        seg_tab = self.seg_embed(torch.full((B, self.n_tab_tokens), 1, dtype=torch.long, device=x_cont.device))
        seg_img = self.seg_embed(torch.full((B, self.n_img_tokens), 2, dtype=torch.long, device=x_cont.device))

        tokens = torch.cat([
            cls + seg_cls,
            tab_tok + seg_tab,
            img_tok + seg_img
        ], dim=1)

        cls_m = torch.zeros(B, 1, dtype=torch.bool, device=x_cont.device)
        tab_m = torch.zeros(B, self.n_tab_tokens, dtype=torch.bool, device=x_cont.device)
        img_m = img_pad_mask.unsqueeze(-1)\
                    .expand(B, self.max_images, self.n_patches_per_img)\
                    .reshape(B, self.n_img_tokens)
        kp_mask = torch.cat([cls_m, tab_m, img_m], dim=1)
        return tokens, kp_mask

    def forward(self, x_cont, x_cat, images, img_pad_mask, mask_bool=None):
        tokens, kp_mask = self._tokenize(x_cont, x_cat, images, img_pad_mask)

        if mask_bool is not None:
            original = tokens.clone()
            tokens = tokens.masked_fill(mask_bool.unsqueeze(-1), 0.0)

        encoded = self.transformer(tokens, src_key_padding_mask=kp_mask)
        encoded = self.norm(encoded)

        if mask_bool is not None:
            return encoded, original, kp_mask

        cls_out = encoded[:, 0]
        return self.classifier(cls_out)

# ======================================================================
# 5. PRE-TRAINING
# ======================================================================

def _make_mask(tokens, kp_mask, mask_ratio):
    B, L, _ = tokens.shape
    valid       = ~kp_mask
    valid[:, 0] = False
    rand        = torch.rand(B, L, device=tokens.device)
    rand[~valid] = 2.0
    return rand < mask_ratio


def pretrain_step(model, batch, optimizer):
    x_cont, x_cat, images, pad_mask, _ = [b.to(DEVICE) for b in batch]
    with torch.no_grad():
        tokens, kp_mask = model._tokenize(x_cont, x_cat, images, pad_mask)
    mask = _make_mask(tokens, kp_mask, MASK_RATIO)
    encoded, original, _ = model(x_cont, x_cat, images, pad_mask, mask_bool=mask)
    loss = F.mse_loss(encoded[mask], original[mask].detach())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# ======================================================================
# 6. FINE-TUNING & EVALUATION
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
        logits   = model(x_cont, x_cat, images, pad_mask)
        tot_loss += criterion(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits  = torch.cat(all_logits)
    labels  = torch.cat(all_labels).numpy()
    probs   = F.softmax(logits, dim=1)[:, 1].numpy()
    preds   = logits.argmax(1).numpy()
    avg_loss = tot_loss / len(labels)
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
    fig.suptitle(
        'Kết Quả Huấn Luyện — IMT-Only Multimodal Transformer\n'
        '(Chỉ dùng 1 ảnh IMT cho cả 2 nhãn)',
        fontsize=13, fontweight='bold'
    )
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


def plot_cm(y_true, y_pred, title_suffix='IMT-Only'):
    cm = confusion_matrix(y_true, y_pred)
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
    ax.set_title(f'Ma Trận Nhầm Lẫn — Tập Test\n({title_suffix})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu confusion matrix: {CM_PATH}")


def plot_comparison(imt_results: dict):
    """
    Vẽ biểu đồ so sánh 2 mô hình.
    imt_results = {'acc':..., 'f1':..., 'roc':..., 'prec':..., 'rec':...}
    """
    # Kết quả mô hình gốc (từ log đã biết)
    original = {'acc': 1.0, 'f1': 1.0, 'roc': 1.0, 'prec': 1.0, 'rec': 1.0}
    imt      = imt_results

    metrics      = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    orig_vals    = [original['acc'], original['prec'], original['rec'],
                    original['f1'],  original['roc']]
    imt_vals     = [imt['acc'], imt['prec'], imt['rec'], imt['f1'], imt['roc']]

    x    = np.arange(len(metrics))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, orig_vals, w, label='Gốc (5 ảnh, tối đa)',
                   color='#2196F3', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, imt_vals,  w, label='IMT-Only (1 ảnh IMT)',
                   color='#FF5722', alpha=0.85, edgecolor='white')

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Giá trị', fontsize=12)
    ax.set_title(
        'So Sánh: Mô Hình Gốc (≤5 ảnh) vs IMT-Only (1 ảnh IMT)\n'
        'Ablation Study — Đánh giá đóng góp của ảnh CCA',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, color='#1565C0', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, color='#BF360C', fontweight='bold')

    plt.tight_layout()
    plt.savefig(COMPARE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu biểu đồ so sánh: {COMPARE_PATH}")

# ======================================================================
# 8. MAIN
# ======================================================================

def main():
    bar = "=" * 64
    logger.info(bar)
    logger.info("  IMT-ONLY MULTIMODAL TRANSFORMER — ABLATION STUDY")
    logger.info("  Chỉ dùng 1 ảnh IMT cho CẢ HAI nhãn (Class 0 & 1)")
    logger.info(bar)
    logger.info(f"Device : {DEVICE}")
    logger.info(f"★ MAX_IMAGES = {MAX_IMAGES} (thay vì 5 của mô hình gốc)")
    logger.info(f"Tokens : tab={N_TAB_TOKENS}, img={N_IMG_TOKENS}, "
                f"total={1+N_TAB_TOKENS+N_IMG_TOKENS} (+CLS)")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data     = load_and_preprocess()
    train_ds = IMTOnlyDataset(data, data['train_idx'])
    test_ds  = IMTOnlyDataset(data, data['test_idx'])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalTransformerIMTOnly(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        n_cont=N_CONT, cat_vocab_sizes=(2,),
        embed_dim=EMBED_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM,
        dropout=DROPOUT, max_images=MAX_IMAGES
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Số tham số mô hình: {n_params:,} "
                f"(so với 584,002 của mô hình gốc)")

    # ── Giai đoạn 1: Pre-training ─────────────────────────────────────
    logger.info("\n" + bar)
    logger.info(" GIAI ĐOẠN 1: TIỀN HUẤN LUYỆN TỰ GIÁM SÁT (SSP)")
    logger.info(f" Mask ratio={MASK_RATIO*100:.0f}% | Epochs={PRETRAIN_EPOCHS}")
    logger.info(bar)

    opt_pre   = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
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
                    f"LR={sched_pre.get_last_lr()[0]:.2e} | T={time.time()-t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PRETRAIN_PATH)
    logger.info(f"\n→ Lưu trọng số tiền huấn luyện: {MODEL_PRETRAIN_PATH}")

    # ── Giai đoạn 2: Fine-tuning ──────────────────────────────────────
    logger.info("\n" + bar)
    logger.info(" GIAI ĐOẠN 2: TINH CHỈNH CÓ GIÁM SÁT")
    logger.info(f" Epochs={FINETUNE_EPOCHS} | LR={LR_FINETUNE}")
    logger.info(bar)

    criterion = nn.CrossEntropyLoss()
    opt_ft    = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    sched_ft  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=FINETUNE_EPOCHS)

    tr_losses, va_losses = [], []
    tr_accs,   va_accs   = [], []
    tr_f1s,    va_f1s    = [], []
    best_f1, best_ep     = -1.0, 1
    t0 = time.time()

    for ep in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        ep_losses, ep_logits, ep_labels = [], [], []
        for batch in train_dl:
            loss, logits, labels = finetune_step(model, batch, opt_ft, criterion)
            ep_losses.append(loss); ep_logits.append(logits); ep_labels.append(labels)
        sched_ft.step()

        all_log = torch.cat(ep_logits)
        all_lbl = torch.cat(ep_labels).numpy()
        all_prd = all_log.argmax(1).numpy()
        tr_loss = np.mean(ep_losses)
        tr_acc  = accuracy_score(all_lbl, all_prd)
        tr_f1   = f1_score(all_lbl, all_prd, zero_division=0)

        va_loss, va_acc, va_prec, va_rec, va_f1, va_roc, *_ = evaluate(
            model, test_dl, criterion
        )
        model.train()

        tr_losses.append(tr_loss); va_losses.append(va_loss)
        tr_accs.append(tr_acc);    va_accs.append(va_acc)
        tr_f1s.append(tr_f1);      va_f1s.append(va_f1)

        flag = ''
        if va_f1 > best_f1:
            best_f1, best_ep = va_f1, ep
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'val_f1': va_f1, 'val_roc': va_roc, 'val_acc': va_acc
            }, MODEL_BEST_PATH)
            flag = '  ← BEST ✓'

        logger.info(
            f"[FT] Ep {ep:3d}/{FINETUNE_EPOCHS} | "
            f"Tr Loss={tr_loss:.4f} Acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"Va Loss={va_loss:.4f} Acc={va_acc:.4f} F1={va_f1:.4f} "
            f"AUC={va_roc:.4f} | T={time.time()-t0:.0f}s{flag}"
        )

    # ── Đánh giá cuối ─────────────────────────────────────────────────
    logger.info("\n" + bar)
    logger.info(" ĐÁNH GIÁ CUỐI CÙNG — TẬP TEST")
    logger.info(bar)

    ckpt = torch.load(MODEL_BEST_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Tải trọng số từ epoch {ckpt['epoch']}")

    te_loss, te_acc, te_prec, te_rec, te_f1, te_roc, \
        te_preds, te_labels, _ = evaluate(model, test_dl, criterion)

    logger.info(f"\n{'─'*40}")
    logger.info(f"  [IMT-Only] Accuracy  : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    logger.info(f"  [IMT-Only] Precision : {te_prec:.4f}")
    logger.info(f"  [IMT-Only] Recall    : {te_rec:.4f}")
    logger.info(f"  [IMT-Only] F1-Score  : {te_f1:.4f}")
    logger.info(f"  [IMT-Only] ROC-AUC   : {te_roc:.4f}")
    logger.info(f"  [IMT-Only] Loss      : {te_loss:.4f}")
    logger.info(f"\n  [Mô hình gốc]  Accuracy=1.0 | F1=1.0 | ROC-AUC=1.0")
    logger.info(f"  → Chênh lệch Accuracy : {(1.0 - te_acc)*100:.2f}%")
    logger.info(f"  → Chênh lệch F1       : {1.0 - te_f1:.4f}")
    logger.info(f"  → Chênh lệch ROC-AUC  : {1.0 - te_roc:.4f}")
    logger.info(f"{'─'*40}\n")

    # ── Biểu đồ ───────────────────────────────────────────────────────
    plot_curves(pre_losses, tr_losses, va_losses,
                tr_accs, va_accs, tr_f1s, va_f1s)
    plot_cm(te_labels, te_preds, title_suffix='IMT-Only (1 ảnh)')
    plot_comparison({'acc': te_acc, 'prec': te_prec, 'rec': te_rec,
                     'f1': te_f1,  'roc':  te_roc})

    logger.info(bar)
    logger.info(" HOÀN THÀNH!")
    logger.info(f"  Output dir: {OUTPUT_DIR}")
    logger.info(bar)


if __name__ == '__main__':
    main()
