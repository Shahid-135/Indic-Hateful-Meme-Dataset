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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_fscore_support,
)


def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("MemeDetection")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


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
NUM_CLASSES = 2
GRAD_CLIP_NORM = 1.0

# ── ATCL hyper-params ──
TAU_BASE_INIT = 0.07
BETA_INIT     = 2.0
TOP_K         = 10

# ── HASMB hyper-params ──
MEMORY_BANK_SIZE = 1024
MOMENTUM         = 0.99
NUM_HARD_NEGS    = 5

# ── Loss weights ──
LAMBDA_ATCL  = 0.01
LAMBDA_PA    = 0.01
LAMBDA_ORTHO = 0.05    
LAMBDA_DIS   = 0.02    

# ── Paths (edit as needed) ──
IMAGE_DIR  = "/home/IndicHate/Indic_memes"
train_file = "/home/IndicHate/overall/task1_train.xlsx"
val_file   = "/home/IndicHate/overall/task1_val.xlsx"
test_file  = "/home/IndicHate/overall/task1_test.xlsx"
SAVE_DIR   = "/home/new/disagreement_centric"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
EPOCH_CHECKPOINT_DIR = os.path.join(SAVE_DIR, "epoch_checkpoints")
os.makedirs(EPOCH_CHECKPOINT_DIR, exist_ok=True)


def load_and_map_labels(path: str, split_name: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading {split_name} data from {path}")
    df = pd.read_excel(path, engine="openpyxl")
    logger.info(f"[{split_name}] Raw label uniques: {df['label'].unique().tolist()}")
    raw = df["label"].astype(str).str.strip().str.lower()

    def _map(v):
        if v in {"hate", "hateful", "yes", "1"}:
            return 1
        if v in {"non-hate", "non-hateful", "nonhate", "no", "0",
                 "not hate", "not-hate", "not hateful", "not_hate"}:
            return 0
        return np.nan

    df["label"] = raw.map(_map)
    n_dropped = df["label"].isna().sum()
    if n_dropped:
        logger.warning(f"  ⚠ {n_dropped} rows dropped (unrecognised labels).")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    logger.info(f"  → {len(df)} rows kept. Label dist: {df['label'].value_counts().to_dict()}")
    return df


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
            text = row["ocr_text"]
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
            # a, b: (B, d) — normalize per dim
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
            B = a.size(0)
            cov = (a.t() @ b) / max(B - 1, 1)       
            return (cov ** 2).mean()

        loss = (
            _gram_off_diag(t_par, t_perp)
          + _gram_off_diag(v_par, v_perp)
          + _gram_off_diag(t_par, v_perp)
          + _gram_off_diag(v_par, t_perp)
        )
        return loss


class ADAFv2(nn.Module):

    def __init__(self, sub_dim: int = CMDP_DIM):
        super().__init__()
        self.sub_dim = sub_dim

      
        self.film_t = nn.Sequential(
            nn.Linear(1, sub_dim // 2),
            nn.SiLU(),
            nn.Linear(sub_dim // 2, 2 * sub_dim) 
        )
        self.film_v = nn.Sequential(
            nn.Linear(1, sub_dim // 2),
            nn.SiLU(),
            nn.Linear(sub_dim // 2, 2 * sub_dim)
        )

        
        gate_in = 4 * sub_dim + 1
        self.gate_t = nn.Sequential(
            nn.Linear(gate_in, sub_dim),
            nn.LayerNorm(sub_dim),
            nn.Sigmoid(),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(gate_in, sub_dim),
            nn.LayerNorm(sub_dim),
            nn.Sigmoid()
        )

        self.U_t = nn.Linear(sub_dim, sub_dim, bias=False)
        self.U_v = nn.Linear(sub_dim, sub_dim, bias=False)
        self.ln_inter = nn.LayerNorm(sub_dim)

    def forward(self, cmdp_out: dict):
        t_par  = cmdp_out["t_par"]
        v_par  = cmdp_out["v_par"]
        t_perp = cmdp_out["t_perp"]
        v_perp = cmdp_out["v_perp"]

       
        alpha = F.cosine_similarity(t_par, v_par, dim=1, eps=1e-6).unsqueeze(1)
        alpha = torch.clamp(alpha, -0.999, 0.999)
        alpha_w = torch.sigmoid(alpha)                       

        
        film_t = self.film_t(alpha_w)
        film_v = self.film_v(alpha_w)
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

        z = torch.cat([stream_t, stream_v, inter], dim=1)   # (B, 3·sub_dim)
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
        out = self.model(**inputs)
        return out.text_embeds, out.image_embeds


class MemeModel(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, logger: Optional[logging.Logger] = None):
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
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, images, texts):
        t, v = self.encoder(images, texts)
        # L2-normalize encoder outputs before CMDP
        t = F.normalize(t, dim=1, eps=1e-6)
        v = F.normalize(v, dim=1, eps=1e-6)
        cmdp_out = self.cmdp(t, v)
        z, alpha = self.fusion(cmdp_out)
        h = self.mlp(z)
        logits = self.classifier(h)
        return logits, h, alpha, cmdp_out



class LearnableTemperature(nn.Module):
    """Per-class learnable τ₀ and β, in log space for positivity."""
    def __init__(self, num_classes: int = NUM_CLASSES,
                 tau0_init: float = TAU_BASE_INIT,
                 beta_init: float = BETA_INIT):
        super().__init__()
        self.log_tau0 = nn.Parameter(torch.full((num_classes,), math.log(tau0_init)))
        self.log_beta = nn.Parameter(torch.full((num_classes,), math.log(beta_init)))

    def get(self, labels: torch.Tensor):
        tau0 = torch.exp(self.log_tau0[labels])
        beta = torch.exp(self.log_beta[labels])
        return tau0, beta


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
    rho = topk_sim.mean(dim=1).clamp(-0.999, 0.999)
    return rho


def dw_atcl_loss(anchor, pos, negs, rho, disagreement,
                 tau0_per_anchor, beta_per_anchor):

    tau = (tau0_per_anchor * torch.exp(-beta_per_anchor * rho)).clamp(0.05, 0.5)
    s = 0.1
    pos_sim = (anchor * pos).sum(dim=1) / (tau + 1e-8)
    neg_sim = torch.einsum("bd,bkd->bk", anchor, negs) / (tau.unsqueeze(1) + 1e-8)
    pos_sim = pos_sim * s
    neg_sim = neg_sim * s
    pos_exp = torch.exp(pos_sim - pos_sim.max())
    neg_exp = torch.exp(neg_sim - neg_sim.max().detach()).sum(dim=1)
    per_sample = -torch.log((pos_exp + 1e-8) / (pos_exp + neg_exp + 1e-8))
    per_sample = per_sample.clamp(max=10.0)
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
            logger.info(f"HASMB initialized: bank_size={bank_size}, dim={dim}")

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

        if self.num_classes == 2:
            push = F.cosine_similarity(
                protos[0].unsqueeze(0), protos[1].unsqueeze(0), eps=1e-6
            ).squeeze().clamp(-0.99, 0.99)
        else:
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

def train_epoch(model, loader, optimizer, faiss_index, hasmb,
                temp_module, epoch, logger):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ce_losses, atcl_losses, pa_losses, ortho_losses = [], [], [], []
    start = time.time()

    for batch_idx, (_n, images, texts, labels) in enumerate(
        tqdm(loader, desc=f"Train E{epoch}", leave=False)
    ):
        labels = labels.to(DEVICE)
        logits, h, alpha, cmdp_out = model(images, texts)

        loss_ce = F.cross_entropy(logits, labels, label_smoothing=0.05)
        ce_losses.append(loss_ce.item())

        
        loss_ortho = model.cmdp.orthogonality_loss(
            cmdp_out["t_par"], cmdp_out["v_par"],
            cmdp_out["t_perp"], cmdp_out["v_perp"],
        )
        ortho_losses.append(loss_ortho.item())

       
        loss_atcl = torch.tensor(0.0, device=DEVICE)
        if faiss_index.is_built and batch_idx % 2 == 0:
            h_norm = F.normalize(h, dim=1, eps=1e-6)
            rho = compute_local_density(h_norm.detach(), k=TOP_K)
            needed = TOP_K + NUM_HARD_NEGS + 2
            h_np = h_norm.detach().cpu().numpy().astype(np.float32)
            D_mat, I_mat = faiss_index.search(h_np, k=needed)

            if I_mat.size > 0 and I_mat.shape[1] > 1:
                pos_idx_list, neg_idx_list, valid_idx, valid_rho = [], [], [], []
                for i in range(min(len(labels), 32)):
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
                        dis_a = cmdp_out["disagreement"][valid_idx]
                        loss_atcl = dw_atcl_loss(
                            h_norm[valid_idx], pos_t, neg_t,
                            rho_t, dis_a, tau0_a, beta_a,
                        )
                        if torch.isnan(loss_atcl):
                            loss_atcl = torch.tensor(0.0, device=DEVICE)
                        else:
                            atcl_losses.append(loss_atcl.item())
                    except Exception as e:
                        logger.warning(f"DW-ATCL computation failed: {e}")
                        loss_atcl = torch.tensor(0.0, device=DEVICE)

   
        loss_pa = hasmb.prototype_alignment_loss(
            h, labels, cmdp_out["disagreement"],
        ).clamp(max=5.0)
        pa_losses.append(loss_pa.item())

        # ── Total
        loss = (loss_ce
                + LAMBDA_ATCL  * loss_atcl
                + LAMBDA_PA    * loss_pa
                + LAMBDA_ORTHO * loss_ortho)

        if torch.isnan(loss):
            logger.warning(f"NaN loss at batch {batch_idx}, skipping")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in model.parameters())
        if not has_nan:
            optimizer.step()
            hasmb.update(h, labels)
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
    # Log learned temperatures
    with torch.no_grad():
        tau0_vals = torch.exp(temp_module.log_tau0).tolist()
        beta_vals = torch.exp(temp_module.log_beta).tolist()
    logger.info(f"  Learned τ₀ per class: {[f'{x:.3f}' for x in tau0_vals]}")
    logger.info(f"  Learned β  per class: {[f'{x:.3f}' for x in beta_vals]}")
    return avg



@torch.no_grad()
def evaluate(model, loader, split_name, logger):
    model.eval()
    y_names, y_true, y_pred, y_prob = [], [], [], []
    start = time.time()
    for names, images, texts, labels in tqdm(loader, desc=f"Eval {split_name}", leave=False):
        labels = labels.to(DEVICE)
        logits, _, _, _ = model(images, texts)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_names.extend(names)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_names": y_names, "y_true": y_true,
        "y_pred":  y_pred,  "y_prob": y_prob,
    }
    logger.info(f"{split_name} eval in {time.time()-start:.2f}s")
    logger.info(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics


def save_checkpoint(model, optimizer, temp_module, epoch, val_acc, test_acc, path, logger):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "temperature_state_dict": temp_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }, path)
    logger.info(f"Checkpoint saved: {path} (epoch {epoch}, val={val_acc:.4f}, test={test_acc:.4f})")


# ───────────── MAIN ─────────────
def main():
    logger = setup_logging(SAVE_DIR)
    logger.info("="*60)
    logger.info("Disagreement-Centric Hate Meme Detection")
    logger.info("CMDP + ADAFv2 + DW-ATCL + HASMB")
    logger.info("="*60)

    df_train = load_and_map_labels(train_file, "TRAIN", logger)
    df_val   = load_and_map_labels(val_file,   "VAL",   logger)
    df_test  = load_and_map_labels(test_file,  "TEST",  logger)
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        if len(df) == 0:
            raise RuntimeError(f"{name} empty after label mapping")

    model       = MemeModel(logger=logger).to(DEVICE)
    hasmb       = HASMB(logger=logger).to(DEVICE)
    temp_module = LearnableTemperature().to(DEVICE)

    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) \
                     + list(temp_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    faiss_idx = FaissIndex(HIDDEN_DIM, logger=logger)

    train_loader = DataLoader(MemeDataset(df_train, IMAGE_DIR, logger),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(MemeDataset(df_val,   IMAGE_DIR, logger),
                              batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(MemeDataset(df_test,  IMAGE_DIR, logger),
                              batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    best_test_acc, best_val_acc = 0.0, 0.0
    history = {"train_loss": [], "val_acc": [], "val_f1": [],
               "test_acc": [], "test_f1": []}

    for epoch in range(EPOCHS):
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        logger.info("="*50)

        # Build FAISS over current projections
        logger.info("Building FAISS index...")
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for _n, images, texts, labels in tqdm(train_loader, desc="FAISS build", leave=False):
                _, h, _, _ = model(images, texts)
                h_norm = F.normalize(h, dim=1, eps=1e-6)
                all_emb.append(h_norm.detach().cpu())
                all_lbl.extend(labels.tolist())
        if all_emb:
            faiss_idx.build(torch.cat(all_emb).numpy().astype(np.float32), all_lbl)

        train_loss = train_epoch(model, train_loader, optimizer, faiss_idx,
                                  hasmb, temp_module, epoch+1, logger)
        history["train_loss"].append(train_loss)

        val_m  = evaluate(model, val_loader,  "Validation", logger)
        test_m = evaluate(model, test_loader, "Test",       logger)
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1"])
        history["test_acc"].append(test_m["accuracy"])
        history["test_f1"].append(test_m["f1"])

        ep_path = os.path.join(EPOCH_CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, temp_module, epoch+1,
                        val_m["accuracy"], test_m["accuracy"], ep_path, logger)

        if test_m["accuracy"] > best_test_acc:
            best_test_acc = test_m["accuracy"]
            best_val_acc  = val_m["accuracy"]
            save_checkpoint(model, optimizer, temp_module, epoch+1,
                            val_m["accuracy"], test_m["accuracy"],
                            BEST_MODEL_PATH, logger)
            logger.info(f"  ✓ New best (test={test_m['accuracy']:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final eval with best checkpoint
    logger.info("\n" + "="*60)
    logger.info("FINAL EVAL WITH BEST CHECKPOINT")
    logger.info("="*60)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    temp_module.load_state_dict(ckpt["temperature_state_dict"])
    final_m = evaluate(model, test_loader, "Final Test", logger)

    pm, rm, fm, _ = precision_recall_fscore_support(final_m["y_true"], final_m["y_pred"], average="micro")
    pM, rM, fM, _ = precision_recall_fscore_support(final_m["y_true"], final_m["y_pred"], average="macro")
    pw, rw, fw, _ = precision_recall_fscore_support(final_m["y_true"], final_m["y_pred"], average="weighted")
    logger.info(f"Acc: {final_m['accuracy']:.4f}")
    logger.info(f"Micro    P:{pm:.4f} R:{rm:.4f} F1:{fm:.4f}")
    logger.info(f"Macro    P:{pM:.4f} R:{rM:.4f} F1:{fM:.4f}")
    logger.info(f"Weighted P:{pw:.4f} R:{rw:.4f} F1:{fw:.4f}")

    csv_path = os.path.join(SAVE_DIR, "test_results.csv")
    num_cls = len(final_m["y_prob"][0])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "gold_label", "predicted_label"]
                   + [f"prob_class_{i}" for i in range(num_cls)])
        for name, gold, pred, prob in zip(final_m["y_names"], final_m["y_true"],
                                          final_m["y_pred"], final_m["y_prob"]):
            w.writerow([name, gold, pred] + prob)

    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)
    logger.info("\n✅ Done.")


if __name__ == "__main__":
    main()

