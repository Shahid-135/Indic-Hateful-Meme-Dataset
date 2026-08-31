#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import gc
import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


# ───────────────────────── LOGGING ─────────────────────────
def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("MemeDetection")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ───────────────────────── CONFIG ─────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "google/siglip2-base-patch16-224"
EMB_DIM     = 768
CMDP_DIM    = 512
FUSED_DIM   = 3 * CMDP_DIM
HIDDEN_DIM  = 512

BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 5  # TASK 2: Religious, Political, Gender Based, Racism, Others
GRAD_CLIP_NORM = 1.0

# ── Scheduler ──
WARMUP_EPOCHS = 3
MIN_LR_RATIO  = 0.01

# ── Early stopping ──
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4

# ── ATCL hyper-params ──
TAU_BASE_INIT = 0.1
BETA_INIT     = 1.0
TOP_K         = 15

# ── HASMB hyper-params ──
MEMORY_BANK_SIZE = 1024
MOMENTUM         = 0.99
NUM_HARD_NEGS    = 8

# ── Loss weights ──
LAMBDA_ATCL  = 0.3
LAMBDA_PA    = 0.2
LAMBDA_ORTHO = 0.05

# ── Sampler ──
USE_BALANCED_SAMPLER = False

# ── AMP ──
USE_AMP = True

# ── Paths for Task 2 ──
IMAGE_DIR  = "/home/IndicHate/Indic_memes"
train_file = "/home/IndicHate/overall/task2_train.xlsx"
val_file   = "/home/IndicHate/overall/task2_val.xlsx"
test_file  = "/home/IndicHate/overall/task2_test.xlsx"
SAVE_DIR   = "/homeL/new/disagreement_centric_task2"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
EPOCH_CHECKPOINT_DIR = os.path.join(SAVE_DIR, "epoch_checkpoints")
os.makedirs(EPOCH_CHECKPOINT_DIR, exist_ok=True)


def load_and_map_categories(path: str, split_name: str, logger: logging.Logger, 
                            label_encoder: LabelEncoder = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load Excel for Task 2, map Hate_Category to class indices."""
    logger.info(f"Loading {split_name} data from {path}")
    df = pd.read_excel(path, engine="openpyxl")
    
    if 'Hate_Category' not in df.columns:
        logger.error(f"'Hate_Category' column not found in {split_name}. Available: {df.columns.tolist()}")
        raise KeyError(f"'Hate_Category' column not found")
    
    unique_categories = df['Hate_Category'].unique().tolist()
    logger.info(f"[{split_name}] Raw categories: {unique_categories}")
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Hate_Category'])
        logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        df['label'] = df['Hate_Category'].apply(
            lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
        )
        n_dropped = (df['label'] == -1).sum()
        if n_dropped > 0:
            logger.warning(f"  ⚠ {n_dropped} rows with unseen categories dropped")
            df = df[df['label'] != -1]
    
    df['label'] = df['label'].astype(int)
    
    label_dist = df['label'].value_counts().to_dict()
    label_dist_named = {label_encoder.classes_[k]: v for k, v in label_dist.items()}
    logger.info(f"  → {len(df)} rows kept. Label dist: {label_dist_named}")
    
    return df, label_encoder


class MemeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, logger: Optional[logging.Logger] = None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.logger    = logger

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, str(row["image_name"]))
            img = Image.open(img_path).convert("RGB")
            text = row.get("ocr_text", "")
            if not isinstance(text, str) or text.strip().lower() in {"nan", ""}:
                text = "a sample text"
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return str(row["image_name"]), img, text, label
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading sample {idx}: {e}")
            dummy = Image.new("RGB", (224, 224), color="black")
            return f"error_{idx}", dummy, "sample text", torch.tensor(0, dtype=torch.long)


def collate_fn(batch):
    names  = [b[0] for b in batch]
    images = [b[1] for b in batch]
    texts  = [b[2] for b in batch]
    labels = torch.stack([b[3] for b in batch])
    return names, images, texts, labels


def make_balanced_sampler(df: pd.DataFrame, label_encoder: LabelEncoder) -> WeightedRandomSampler:
    """Per-class inverse-frequency sampling weights."""
    counts = df["label"].value_counts().to_dict()
    sample_weights = df["label"].map(lambda y: 1.0 / counts[y]).values
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )



class CMDP(nn.Module):
    def __init__(self, in_dim: int = EMB_DIM, sub_dim: int = CMDP_DIM):
        super().__init__()
        self.sub_dim = sub_dim
        self.W_agree = nn.Linear(in_dim, sub_dim, bias=False)
        self.W_text_perp   = nn.Linear(in_dim, sub_dim, bias=False)
        self.W_vision_perp = nn.Linear(in_dim, sub_dim, bias=False)
        self.ln_agree_t = nn.LayerNorm(sub_dim)
        self.ln_agree_v = nn.LayerNorm(sub_dim)
        self.ln_perp_t  = nn.LayerNorm(sub_dim)
        self.ln_perp_v  = nn.LayerNorm(sub_dim)

    def forward(self, t: torch.Tensor, v: torch.Tensor):
        t_par = self.ln_agree_t(self.W_agree(t))
        v_par = self.ln_agree_v(self.W_agree(v))
        t_perp = self.ln_perp_t(self.W_text_perp(t))
        v_perp = self.ln_perp_v(self.W_vision_perp(v))

        d_text   = torch.norm(t_perp, dim=1) / math.sqrt(self.sub_dim)
        d_vision = torch.norm(v_perp, dim=1) / math.sqrt(self.sub_dim)
        disagreement = torch.tanh(0.5 * (d_text + d_vision))

        return {
            "t_par": t_par,    "v_par": v_par,
            "t_perp": t_perp,  "v_perp": v_perp,
            "disagreement": disagreement,
        }

    def orthogonality_loss(self, t_par, v_par, t_perp, v_perp):
        def _gram_off_diag(a, b):
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
            B = a.size(0)
            cov = (a.t() @ b) / max(B - 1, 1)
            return (cov ** 2).mean()

        return (
            _gram_off_diag(t_par, t_perp)
          + _gram_off_diag(v_par, v_perp)
          + _gram_off_diag(t_par, v_perp)
          + _gram_off_diag(v_par, t_perp)
        )


class ADAFv2(nn.Module):
    def __init__(self, sub_dim: int = CMDP_DIM):
        super().__init__()
        self.sub_dim = sub_dim
        self.film_t = nn.Sequential(
            nn.Linear(1, sub_dim // 2), nn.SiLU(),
            nn.Linear(sub_dim // 2, 2 * sub_dim),
        )
        self.film_v = nn.Sequential(
            nn.Linear(1, sub_dim // 2), nn.SiLU(),
            nn.Linear(sub_dim // 2, 2 * sub_dim),
        )
        gate_in = 4 * sub_dim + 1
        self.gate_t = nn.Sequential(
            nn.Linear(gate_in, sub_dim), nn.LayerNorm(sub_dim), nn.Sigmoid(),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(gate_in, sub_dim), nn.LayerNorm(sub_dim), nn.Sigmoid(),
        )
        self.U_t = nn.Linear(sub_dim, sub_dim, bias=False)
        self.U_v = nn.Linear(sub_dim, sub_dim, bias=False)
        self.ln_inter = nn.LayerNorm(sub_dim)

    def forward(self, cmdp_out: dict):
        t_par  = cmdp_out["t_par"]; v_par  = cmdp_out["v_par"]
        t_perp = cmdp_out["t_perp"]; v_perp = cmdp_out["v_perp"]

        alpha = F.cosine_similarity(t_par, v_par, dim=1, eps=1e-6).unsqueeze(1)
        alpha = torch.clamp(alpha, -0.999, 0.999)
        alpha_w = torch.sigmoid(alpha)

        film_t = self.film_t(alpha_w); film_v = self.film_v(alpha_w)
        gamma_t, beta_t = film_t.chunk(2, dim=1)
        gamma_v, beta_v = film_v.chunk(2, dim=1)
        gamma_t = 1.0 + gamma_t
        gamma_v = 1.0 + gamma_v

        gate_in_t = torch.cat([t_par, v_par, t_perp, v_perp, alpha_w], dim=1)
        gate_in_v = torch.cat([v_par, t_par, v_perp, t_perp, alpha_w], dim=1)
        g_t = self.gate_t(gate_in_t)
        g_v = self.gate_v(gate_in_v)

        stream_t = g_t * (gamma_t * (t_par + t_perp) + beta_t)
        stream_v = g_v * (gamma_v * (v_par + v_perp) + beta_v)

        inter = self.ln_inter(self.U_t(t_par) * self.U_v(v_par))

        z = torch.cat([stream_t, stream_v, inter], dim=1)
        return z, alpha_w.squeeze(1)


class SigLIP2Encoder(nn.Module):
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger
        if logger: logger.info(f"Loading SigLIP2 model: {MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        for p in self.model.parameters():
            p.requires_grad = False
        if logger: logger.info("SigLIP2 encoder loaded and frozen")

    def forward(self, images, texts):
        inputs = self.processor(
            images=images, text=texts,
            padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        ).to(DEVICE)
        with autocast(device_type="cuda", enabled=False):
            out = self.model(**inputs)
        return out.text_embeds.float(), out.image_embeds.float()



class MemeModel(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES,
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.encoder = SigLIP2Encoder(logger)
        self.cmdp    = CMDP(EMB_DIM, CMDP_DIM)
        self.fusion  = ADAFv2(CMDP_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(FUSED_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, texts):
        t, v = self.encoder(images, texts)
        t = F.normalize(t, dim=1, eps=1e-6)
        v = F.normalize(v, dim=1, eps=1e-6)
        cmdp_out = self.cmdp(t, v)
        z, alpha = self.fusion(cmdp_out)
        h = self.mlp(z)
        logits = self.classifier(h)
        return logits, h, alpha, cmdp_out


class LearnableTemperature(nn.Module):
    """
    Per-class learnable τ₀ and β, in log space for positivity.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 tau0_init: float = TAU_BASE_INIT,
                 beta_init: float = BETA_INIT):
        super().__init__()
        self.log_tau0 = nn.Parameter(torch.full((num_classes,), math.log(tau0_init)))
        self.log_beta = nn.Parameter(torch.full((num_classes,), math.log(beta_init)))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def get(self, labels: torch.Tensor):
        tau0 = torch.exp(self.log_tau0[labels])
        beta = torch.exp(self.log_beta[labels])
        return tau0, beta

    @property
    def scale(self):
        return torch.exp(self.log_scale).clamp(0.5, 10.0)


def compute_local_density(emb_norm: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
    B = emb_norm.size(0)
    k = min(k, B - 1)
    if k <= 0 or B < 2:
        return torch.zeros(B, device=emb_norm.device)
    sim = torch.mm(emb_norm, emb_norm.t())
    mask = torch.eye(B, dtype=torch.bool, device=emb_norm.device)
    sim = sim.masked_fill(mask, -float("inf"))
    topk_sim, _ = torch.topk(sim, k, dim=1)
    topk_sim = torch.where(torch.isinf(topk_sim), torch.zeros_like(topk_sim), topk_sim)
    return topk_sim.mean(dim=1).clamp(-0.999, 0.999)


def dw_atcl_loss(anchor, pos, negs, rho, disagreement,
                 tau0_per_anchor, beta_per_anchor, scale):
    """
    τ_i = clip( τ₀(y_i) * exp(-β(y_i) * ρ_i),  0.05, 0.5 )
    """
    tau = (tau0_per_anchor * torch.exp(-beta_per_anchor * rho)).clamp(0.05, 0.5)
    pos_sim = scale * (anchor * pos).sum(dim=1) / (tau + 1e-8)
    neg_sim = scale * torch.einsum("bd,bkd->bk", anchor, negs) / (tau.unsqueeze(1) + 1e-8)

    all_logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    log_prob = pos_sim - torch.logsumexp(all_logits, dim=1)
    per_sample = (-log_prob).clamp(max=20.0)

    weights = 0.5 + disagreement.detach()
    return (weights * per_sample).mean()


class HASMB(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, bank_size=MEMORY_BANK_SIZE,
                 num_classes=NUM_CLASSES, momentum=MOMENTUM,
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.dim = dim
        self.bank_size = bank_size
        self.num_classes = num_classes
        self.momentum = momentum
        self.logger = logger
        self.register_buffer("bank", torch.randn(num_classes, bank_size, dim) * 0.01)
        self.register_buffer("ptr",  torch.zeros(num_classes, dtype=torch.long))
        if logger:
            logger.info(f"HASMB initialized: bank_size={bank_size}, dim={dim}, num_classes={num_classes}")

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        emb = F.normalize(embeddings.detach(), dim=1, eps=1e-6)
        for c in range(self.num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            batch_c = emb[idx].cpu()
            n = batch_c.size(0)
            ptr = int(self.ptr[c])
            end = min(ptr + n, self.bank_size)
            self.bank[c, ptr:end] = batch_c[:end - ptr]
            if end < ptr + n:
                rem = ptr + n - self.bank_size
                self.bank[c, :rem] = batch_c[end - ptr:]
            self.ptr[c] = (ptr + n) % self.bank_size

    def prototypes(self) -> torch.Tensor:
        bank_mean = self.bank.mean(dim=1)
        bank_mean = torch.where(torch.isnan(bank_mean), torch.zeros_like(bank_mean), bank_mean)
        return F.normalize(bank_mean, dim=1, eps=1e-6)

    def prototype_alignment_loss(self, h, labels, disagreement, lambda_sep: float = 0.5):
        protos = self.prototypes()
        h_norm = F.normalize(h, dim=1, eps=1e-6)
        per_sample = ((h_norm - protos[labels]) ** 2).sum(dim=1)
        weights = 0.5 + disagreement.detach()
        pull = (weights * per_sample).mean().clamp(max=10.0)

        # Multi-class push loss: minimize mean pairwise similarity between different class prototypes
        sim_mat = protos @ protos.t()
        mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=h.device)
        push = sim_mat[mask].mean()
        return pull + lambda_sep * push


class FaissIndex:
    def __init__(self, dim: int, logger: Optional[logging.Logger] = None):
        self.index = faiss.IndexFlatIP(dim)
        self.embeddings = None
        self.labels = []
        self.logger = logger
        self._built = False

    def build(self, embeddings: np.ndarray, labels: list):
        emb = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(emb)
        self.embeddings = emb.copy()
        self.index.reset()
        self.index.add(emb)
        self.labels = labels
        self._built = True
        if self.logger:
            self.logger.info(f"FAISS index built: {self.index.ntotal} vectors")

    def search(self, queries: np.ndarray, k: int):
        if not self.is_built or self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        q = queries.astype(np.float32).copy()
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        if k == 0:
            return np.array([[]]), np.array([[]])
        return self.index.search(q, k)

    def reconstruct_batch(self, indices: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("Index not built yet")
        return self.embeddings[indices]

    @property
    def is_built(self) -> bool:
        return self._built and self.index.ntotal > 0


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopper:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE,
                 min_delta: float = EARLY_STOP_MIN_DELTA, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.best_epoch = -1

    def step(self, value: float, epoch: int):
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
                   else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            self.best_epoch = epoch
            return True
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


def train_epoch(model, loader, optimizer, scheduler, scaler,
                faiss_index, hasmb, temp_module, epoch, logger):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ce_losses, atcl_losses, pa_losses, ortho_losses = [], [], [], []
    start = time.time()

    for batch_idx, (_n, images, texts, labels) in enumerate(
        tqdm(loader, desc=f"Train E{epoch}", leave=False)
    ):
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE == "cuda")):
            logits, h, alpha, cmdp_out = model(images, texts)
            loss_ce = F.cross_entropy(logits, labels, label_smoothing=0.05)

        loss_ortho = model.cmdp.orthogonality_loss(
            cmdp_out["t_par"].float(), cmdp_out["v_par"].float(),
            cmdp_out["t_perp"].float(), cmdp_out["v_perp"].float(),
        )
        ortho_losses.append(loss_ortho.item())
        ce_losses.append(loss_ce.item())

        # DW-ATCL
        loss_atcl = torch.tensor(0.0, device=DEVICE)
        if faiss_index.is_built:
            h_norm = F.normalize(h.float(), dim=1, eps=1e-6)
            rho = compute_local_density(h_norm.detach(), k=TOP_K)
            needed = TOP_K + NUM_HARD_NEGS + 2
            h_np = h_norm.detach().cpu().numpy().astype(np.float32)
            D_mat, I_mat = faiss_index.search(h_np, k=needed)

            if I_mat.size > 0 and I_mat.shape[1] > 1:
                pos_idx_list, neg_idx_list, valid_idx, valid_rho = [], [], [], []
                for i in range(len(labels)):
                    al = labels[i].item()
                    pos_i, neg_i = None, []
                    for j in range(1, min(I_mat.shape[1], needed)):
                        raw = int(I_mat[i][j])
                        if raw < 0 or raw >= len(faiss_index.labels):
                            continue
                        lbl = faiss_index.labels[raw]
                        if lbl == al and pos_i is None:
                            pos_i = raw
                        elif lbl != al and len(neg_i) < NUM_HARD_NEGS:
                            neg_i.append(raw)
                        if pos_i is not None and len(neg_i) == NUM_HARD_NEGS:
                            break
                    if pos_i is not None and len(neg_i) == NUM_HARD_NEGS:
                        pos_idx_list.append(pos_i)
                        neg_idx_list.append(neg_i)
                        valid_idx.append(i)
                        valid_rho.append(rho[i])

                if pos_idx_list:
                    try:
                        pos_np = faiss_index.reconstruct_batch(np.array(pos_idx_list))
                        neg_np = np.array([
                            faiss_index.reconstruct_batch(np.array(neg))
                            for neg in neg_idx_list
                        ])
                        pos_t = F.normalize(torch.tensor(pos_np, dtype=torch.float32, device=DEVICE), dim=1)
                        neg_t = F.normalize(torch.tensor(neg_np, dtype=torch.float32, device=DEVICE), dim=2)
                        rho_t = torch.stack(valid_rho).to(DEVICE)
                        valid_labels = labels[valid_idx]
                        tau0_a, beta_a = temp_module.get(valid_labels)
                        dis_a = cmdp_out["disagreement"][valid_idx].float()
                        loss_atcl = dw_atcl_loss(
                            h_norm[valid_idx], pos_t, neg_t,
                            rho_t, dis_a, tau0_a, beta_a,
                            scale=temp_module.scale,
                        )
                        if torch.isnan(loss_atcl):
                            loss_atcl = torch.tensor(0.0, device=DEVICE)
                        else:
                            atcl_losses.append(loss_atcl.item())
                    except Exception as e:
                        logger.warning(f"DW-ATCL computation failed: {e}")
                        loss_atcl = torch.tensor(0.0, device=DEVICE)

        # Prototype alignment
        loss_pa = hasmb.prototype_alignment_loss(
            h.float(), labels, cmdp_out["disagreement"].float(),
        ).clamp(max=5.0)
        pa_losses.append(loss_pa.item())

        # Total loss
        loss = (loss_ce
                + LAMBDA_ATCL  * loss_atcl
                + LAMBDA_PA    * loss_pa
                + LAMBDA_ORTHO * loss_ortho)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            continue


        if USE_AMP and DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                          for p in model.parameters())
            if not has_nan:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                hasmb.update(h.float(), labels)
                total_loss += loss.item()
                n_batches += 1
            else:
                scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                          for p in model.parameters())
            if not has_nan:
                optimizer.step()
                scheduler.step()
                hasmb.update(h.float(), labels)
                total_loss += loss.item()
                n_batches += 1

    dt = time.time() - start
    avg = total_loss / n_batches if n_batches else 0.0
    logger.info(f"Epoch {epoch} train done in {dt:.2f}s")
    logger.info(
        f"  Avg Loss: {avg:.4f} "
        f"(CE: {np.mean(ce_losses):.4f}, "
        f"ATCL: {np.mean(atcl_losses) if atcl_losses else 0:.4f}, "
        f"PA: {np.mean(pa_losses):.4f}, "
        f"ORTHO: {np.mean(ortho_losses):.4f})"
    )
    with torch.no_grad():
        tau0_vals = torch.exp(temp_module.log_tau0).tolist()
        beta_vals = torch.exp(temp_module.log_beta).tolist()
        scale_val = temp_module.scale.item()
    logger.info(f"  Learned τ₀ per class: {[f'{x:.3f}' for x in tau0_vals]}")
    logger.info(f"  Learned β  per class: {[f'{x:.3f}' for x in beta_vals]}")
    logger.info(f"  Learned global scale s: {scale_val:.3f}")
    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(f"  Current LR: {current_lr:.2e}")
    return avg



def log_classification_report(y_true, y_pred, split_name, logger, class_names):
    """Print per-class P/R/F1 + macro/weighted averages + confusion matrix."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    logger.info(f"\n  ── {split_name} classification report ──")
    for line in report.strip().split("\n"):
        logger.info(f"    {line}")

    logger.info(f"  ── {split_name} confusion matrix ──")
    header = "    " + " ".ljust(15) + "".join(f"{n:>15}" for n in class_names)
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = "    " + class_names[i].ljust(15) + "".join(f"{v:>15d}" for v in row)
        logger.info(row_str)


@torch.no_grad()
def evaluate(model, loader, split_name, logger, class_names, print_report: bool = True):
    model.eval()
    y_names, y_true, y_pred, y_prob = [], [], [], []
    start = time.time()
    for names, images, texts, labels in tqdm(loader, desc=f"Eval {split_name}", leave=False):
        labels = labels.to(DEVICE)
        with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE == "cuda")):
            logits, _, _, _ = model(images, texts)
        probs = F.softmax(logits.float(), dim=1)
        preds = torch.argmax(probs, dim=1)
        y_names.extend(names)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_names": y_names, "y_true": y_true,
        "y_pred":  y_pred,  "y_prob": y_prob,
    }
    
    logger.info(f"{split_name} eval in {time.time()-start:.2f}s")
    logger.info(f"  Acc: {metrics['accuracy']:.4f}, Macro-F1: {metrics['f1_macro']:.4f}, Weighted-F1: {metrics['f1_weighted']:.4f}")
    
    if print_report:
        log_classification_report(y_true, y_pred, split_name, logger, class_names)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, temp_module, epoch,
                    val_acc, val_f1, test_acc, test_f1, path, logger):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "temperature_state_dict": temp_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_accuracy": val_acc, "val_f1": val_f1,
        "test_accuracy": test_acc, "test_f1": test_f1,
    }, path)
    logger.info(f"Checkpoint: {path} (E{epoch}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, test_acc={test_acc:.4f})")


# ───────────── MAIN ─────────────
def main():
    logger = setup_logging(SAVE_DIR)
    logger.info("="*60)
    logger.info("TASK 2: Disagreement-Centric Hate Meme Detection")
    logger.info("Categories: Religious, Political, Gender Based, Racism, Others")
    logger.info("CMDP + ADAFv2 + DW-ATCL + HASMB")
    logger.info("="*60)
    logger.info(f"AMP: {USE_AMP and DEVICE == 'cuda'} | Device: {DEVICE}")
    logger.info(f"LR={LR}, warmup={WARMUP_EPOCHS}ep, min_lr_ratio={MIN_LR_RATIO}")
    logger.info(f"λ_ATCL={LAMBDA_ATCL}, λ_PA={LAMBDA_PA}, λ_orth={LAMBDA_ORTHO}")
    logger.info(f"Early stopping: patience={EARLY_STOP_PATIENCE} on val_f1")
    logger.info(f"Number of classes: {NUM_CLASSES}")

    # Load data for Task 2
    df_train, label_encoder = load_and_map_categories(train_file, "TRAIN", logger)
    df_val, _ = load_and_map_categories(val_file, "VAL", logger, label_encoder)
    df_test, _ = load_and_map_categories(test_file, "TEST", logger, label_encoder)
    
    class_names = list(label_encoder.classes_)
    logger.info(f"\nClass names: {class_names}")
    
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        if len(df) == 0:
            raise RuntimeError(f"{name} empty after label mapping")

    model = MemeModel(num_classes=NUM_CLASSES, logger=logger).to(DEVICE)
    hasmb = HASMB(num_classes=NUM_CLASSES, logger=logger).to(DEVICE)
    temp_module = LearnableTemperature(num_classes=NUM_CLASSES).to(DEVICE)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) \
                     + list(temp_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    faiss_idx = FaissIndex(HIDDEN_DIM, logger=logger)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Data loaders
    if USE_BALANCED_SAMPLER:
        sampler = make_balanced_sampler(df_train, label_encoder)
        train_loader = DataLoader(MemeDataset(df_train, IMAGE_DIR, logger),
                                  batch_size=BATCH_SIZE, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=0)
        logger.info("Balanced sampler enabled.")
    else:
        train_loader = DataLoader(MemeDataset(df_train, IMAGE_DIR, logger),
                                  batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, drop_last=True)
    
    val_loader   = DataLoader(MemeDataset(df_val, IMAGE_DIR, logger),
                              batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, drop_last=False)
    test_loader  = DataLoader(MemeDataset(df_test, IMAGE_DIR, logger),
                              batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, drop_last=False)

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps  = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    scheduler = build_lr_scheduler(optimizer, total_steps, warmup_steps)
    logger.info(f"Scheduler: {warmup_steps} warmup steps, {total_steps} total steps")

    early = EarlyStopper(patience=EARLY_STOP_PATIENCE,
                         min_delta=EARLY_STOP_MIN_DELTA, mode="max")

    history = {"train_loss": [], "val_acc": [], "val_f1": [],
               "test_acc": [], "test_f1": [], "lr": []}

    for epoch in range(EPOCHS):
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        logger.info("="*50)

        # Build FAISS index
        logger.info("Building FAISS index...")
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for _n, images, texts, labels in tqdm(train_loader, desc="FAISS build", leave=False):
                with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE == "cuda")):
                    _, h, _, _ = model(images, texts)
                h_norm = F.normalize(h.float(), dim=1, eps=1e-6)
                all_emb.append(h_norm.detach().cpu())
                all_lbl.extend(labels.tolist())
        if all_emb:
            faiss_idx.build(torch.cat(all_emb).numpy().astype(np.float32), all_lbl)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  scaler, faiss_idx, hasmb, temp_module,
                                  epoch+1, logger)
        history["train_loss"].append(train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        val_m  = evaluate(model, val_loader,  "Validation", logger, class_names)
        test_m = evaluate(model, test_loader, "Test",       logger, class_names, print_report=False)
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1_macro"])
        history["test_acc"].append(test_m["accuracy"])
        history["test_f1"].append(test_m["f1_macro"])

        ep_path = os.path.join(EPOCH_CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, temp_module, epoch+1,
                        val_m["accuracy"], val_m["f1_macro"],
                        test_m["accuracy"], test_m["f1_macro"], ep_path, logger)

        # Select best on VAL F1
        improved = early.step(val_m["f1_macro"], epoch+1)
        if improved:
            save_checkpoint(model, optimizer, scheduler, temp_module, epoch+1,
                            val_m["accuracy"], val_m["f1_macro"],
                            test_m["accuracy"], test_m["f1_macro"],
                            BEST_MODEL_PATH, logger)
            logger.info(f"  ✓ New best val_f1 = {val_m['f1_macro']:.4f} "
                        f"(test_acc={test_m['accuracy']:.4f}, "
                        f"test_f1={test_m['f1_macro']:.4f})")
        else:
            logger.info(f"  No val_f1 improvement ({early.bad_epochs}/"
                        f"{EARLY_STOP_PATIENCE} bad epochs)")

        if early.should_stop:
            logger.info(f"\n⏹ Early stopping at epoch {epoch+1}. "
                        f"Best val_f1={early.best:.4f} at epoch {early.best_epoch}.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final eval with best checkpoint
    logger.info("\n" + "="*60)
    logger.info("FINAL EVAL — best-on-validation checkpoint")
    logger.info("="*60)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    temp_module.load_state_dict(ckpt["temperature_state_dict"])
    logger.info(f"Loaded epoch {ckpt['epoch']} "
                f"(val_acc={ckpt['val_accuracy']:.4f}, val_f1={ckpt['val_f1']:.4f})")
    final_m = evaluate(model, test_loader, "Final Test", logger, class_names)

    # Detailed metrics
    y_true, y_pred = final_m["y_true"], final_m["y_pred"]
    
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    pM, rM, fM, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    logger.info("\n" + "="*55)
    logger.info("FINAL TEST RESULTS SUMMARY")
    logger.info("="*55)
    logger.info(f"  Accuracy   : {final_m['accuracy']:.4f}")
    logger.info(f"  Micro   → P:{pm:.4f} R:{rm:.4f} F1:{fm:.4f}")
    logger.info(f"  Macro   → P:{pM:.4f} R:{rM:.4f} F1:{fM:.4f}")
    logger.info(f"  Weighted→ P:{pw:.4f} R:{rw:.4f} F1:{fw:.4f}")
    logger.info("="*55)

    # Save results
    csv_path = os.path.join(SAVE_DIR, "test_results.csv")
    num_cls = len(final_m["y_prob"][0])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "gold_label", "gold_category", "predicted_label", "predicted_category"]
                   + [f"prob_{class_names[i]}" for i in range(num_cls)])
        for name, gold, pred, prob in zip(final_m["y_names"], final_m["y_true"],
                                          final_m["y_pred"], final_m["y_prob"]):
            w.writerow([name, gold, class_names[gold], pred, class_names[pred]] + prob)

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)
    
    # Save confusion matrix plot
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Task 2')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
    
    logger.info(f"\n✅ Task 2 completed. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()

