#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL PIPELINE (UP TO CREDIBILITY SCORES) — uses parent_asin as item id

Pipeline:
1) Raw JSONL -> user_labels.csv (Ru + label)
2) Merge labels into JSONL -> *_with_labels.jsonl
3) Feature engineering (6 user features; AAD uses parent_asin item means) -> user_features.csv
4) Merge features into JSONL -> *_with_labels_and_features.jsonl
5) Build PyG HeteroData graph (user-item) with memmap edges
6) Train Credibility model (GraphSAGE-style + EWA) and export credibility scores

Outputs (in dataset/):
- user_labels.csv
- Clothing_Shoes_and_Jewelry_with_labels.jsonl
- user_features.csv
- Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl
- graph_pyg_parent_asin/graph_hetero_rates.pt
- graph_pyg_parent_asin/credibility_scores.npy + credibility_scores.csv + cred_model.pt

Install:
  pip install numpy torch torch-geometric
"""

import csv
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# =========================
# PATH FIX (IMPORTANT)
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG (EDIT THESE)
# =========================
RAW_JSONL = DATASET_DIR / "Clothing_Shoes_and_Jewelry.jsonl"

LABELS_CSV = DATASET_DIR / "user_labels.csv"
LABELED_JSONL = DATASET_DIR / "Clothing_Shoes_and_Jewelry_with_labels.jsonl"

FEATURES_CSV = DATASET_DIR / "user_features.csv"
LABELED_FEATURED_JSONL = DATASET_DIR / "Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl"

OUT_DIR = DATASET_DIR / "graph_pyg_parent_asin"

# ✅ IMPORTANT: item id key
ITEM_ID_KEY = "parent_asin"

# Ru labeling rule
HELPFUL_VOTE_THRESHOLD = 5  # helpful if helpful_vote > 5
RU_GENUINE_TH = 0.7
RU_FAKE_TH = 0.3

# Feature engineering
TAU_MS = 24 * 60 * 60 * 1000  # 1-day buckets for burst approx

# Graph edge attrs
EDGE_ATTR_KEYS = ["verified", "rating_align", "rating", "timestamp_norm", "helpful_vote"]
NORMALIZE_TIMESTAMP = True

# User node features (Ru + 6 engineered)
USER_FEATURE_KEYS = [
    "Ru",
    "rating_entropy",
    "extremity_ratio",
    "average_rating_deviation",
    "review_burst_count",
    "lexical_diversity",
    "review_length_discrepancy",
]

LABEL_TO_INT = {"fake": 0, "genuine": 1, "unlabeled": -1}

# Disk/RAM friendly
IDX_DTYPE = np.int32
F_DTYPE = np.float32
PRINT_EVERY = 1_000_000

# Credibility training
DEVICE_PREF = "cuda"     # "cuda" or "cpu"
EPOCHS = 20
BATCH_SIZE = 2048
LR = 1e-3
HIDDEN_DIM = 64

# Skip steps if output already exists
SKIP_IF_EXISTS = True

# =========================
# Helpers
# =========================
token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def tokenize(text: str):
    if not text:
        return []
    return token_re.findall(text.lower())

def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def calc_rating_align(r_ui: float, rbar_i: float):
    # RatingAlign_{u,i} = 1 - |r_ui - rbar_i| / 4 (rating scale 1..5)
    if r_ui is None or rbar_i is None or math.isnan(r_ui) or math.isnan(rbar_i):
        return np.nan
    return 1.0 - (abs(float(r_ui) - float(rbar_i)) / 4.0)

def entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            H -= p * math.log(p)
    return H

def assert_exists(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p} (resolved: {p.resolve()})")

# =========================
# STEP 1: User labeling (Ru)
# =========================
def step1_build_user_labels(raw_jsonl: Path, out_csv: Path):
    if SKIP_IF_EXISTS and out_csv.exists():
        print(f"[SKIP] labels CSV exists: {out_csv}")
        return

    assert_exists(raw_jsonl, "RAW_JSONL")

    total_reviews = defaultdict(int)
    helpful_reviews = defaultdict(int)

    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            u = r.get("user_id")
            if not u:
                continue

            total_reviews[u] += 1
            hv = safe_int(r.get("helpful_vote", 0), default=0)
            if hv is not None and hv > HELPFUL_VOTE_THRESHOLD:
                helpful_reviews[u] += 1

            if i % PRINT_EVERY == 0:
                print(f"[LABEL] Processed {i:,} reviews | users={len(total_reviews):,}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "total_reviews", "helpful_reviews", "Ru", "label"])

        for u, tot in total_reviews.items():
            hel = helpful_reviews[u]
            Ru = hel / tot if tot else 0.0

            if Ru >= RU_GENUINE_TH:
                label = "genuine"
            elif Ru <= RU_FAKE_TH:
                label = "fake"
            else:
                label = "unlabeled"

            writer.writerow([u, tot, hel, Ru, label])

    print(f"[LABEL] Saved: {out_csv}")
    print(f"[LABEL] Total users labeled: {len(total_reviews):,}")

# =========================
# STEP 2: Merge labels into JSONL
# =========================
def step2_merge_labels(raw_jsonl: Path, labels_csv: Path, out_jsonl: Path):
    if SKIP_IF_EXISTS and out_jsonl.exists():
        print(f"[SKIP] labeled JSONL exists: {out_jsonl}")
        return

    assert_exists(raw_jsonl, "RAW_JSONL")
    assert_exists(labels_csv, "LABELS_CSV")

    user2lab = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user2lab[row["user_id"]] = (float(row["Ru"]), row["label"])

    print(f"[LABEL-MERGE] Loaded users: {len(user2lab):,}")

    missing = 0
    written = 0
    with open(raw_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            r = json.loads(line)
            u = r.get("user_id")

            if u in user2lab:
                Ru, label = user2lab[u]
                r["Ru"] = Ru
                r["label"] = label
            else:
                r["Ru"] = None
                r["label"] = None
                missing += 1

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1

            if i % PRINT_EVERY == 0:
                print(f"[LABEL-MERGE] {i:,} rows | missing users: {missing:,}")

    print("[LABEL-MERGE] Output:", out_jsonl)
    print("[LABEL-MERGE] Rows written:", f"{written:,}")
    print("[LABEL-MERGE] Missing label rows:", f"{missing:,}")

# =========================
# STEP 3: Feature engineering (6 features)
# ✅ AAD uses item mean rating by parent_asin
# =========================
def step3_compute_user_features(labeled_jsonl: Path, out_csv: Path):
    if SKIP_IF_EXISTS and out_csv.exists():
        print(f"[SKIP] features CSV exists: {out_csv}")
        return

    assert_exists(labeled_jsonl, "LABELED_JSONL")

    user_n = defaultdict(int)
    user_r = defaultdict(lambda: [0, 0, 0, 0, 0])
    user_extreme = defaultdict(int)
    user_ttr_sum = defaultdict(float)
    user_bucket_cnt = defaultdict(lambda: defaultdict(int))

    user_Ru = {}
    user_label = {}

    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)

    global_len_sum = 0
    global_len_cnt = 0

    # PASS 1
    with open(labeled_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            uid = r.get("user_id")
            item_id = r.get(ITEM_ID_KEY)
            rating = r.get("rating")

            if not uid or not item_id or rating is None:
                continue

            if uid not in user_Ru:
                user_Ru[uid] = r.get("Ru", None)
                user_label[uid] = r.get("label", None)

            ri = int(round(float(rating)))
            ri = 1 if ri < 1 else 5 if ri > 5 else ri

            user_n[uid] += 1
            user_r[uid][ri - 1] += 1
            if ri in (1, 5):
                user_extreme[uid] += 1

            item_sum[item_id] += ri
            item_cnt[item_id] += 1

            text = (r.get("title") or "") + " " + (r.get("text") or "")
            toks = tokenize(text)
            L = len(toks)
            global_len_sum += L
            global_len_cnt += 1
            if L > 0:
                user_ttr_sum[uid] += len(set(toks)) / L

            ts = r.get("timestamp")
            if ts is not None:
                ts_int = safe_int(ts)
                if ts_int is not None:
                    bucket = int(ts_int // TAU_MS)
                    user_bucket_cnt[uid][bucket] += 1

            if i % PRINT_EVERY == 0:
                print(f"[FEAT] PASS1 {i:,} | users={len(user_n):,} items={len(item_cnt):,}")

    global_avg_len = global_len_sum / max(global_len_cnt, 1)
    item_mean = {a: item_sum[a] / item_cnt[a] for a in item_cnt}
    print(f"[FEAT] Item means computed: {len(item_mean):,} using {ITEM_ID_KEY}")
    print(f"[FEAT] Global avg length: {global_avg_len:.4f}")

    # PASS 2
    user_aad_sum = defaultdict(float)
    user_rd_sum = defaultdict(float)

    with open(labeled_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            uid = r.get("user_id")
            item_id = r.get(ITEM_ID_KEY)
            rating = r.get("rating")
            if not uid or not item_id or rating is None:
                continue

            ri = int(round(float(rating)))
            ri = 1 if ri < 1 else 5 if ri > 5 else ri

            rbar = item_mean.get(item_id)
            if rbar is not None:
                user_aad_sum[uid] += abs(ri - rbar)

            text = (r.get("title") or "") + " " + (r.get("text") or "")
            L = len(tokenize(text))
            user_rd_sum[uid] += abs(L - global_avg_len)

            if i % PRINT_EVERY == 0:
                print(f"[FEAT] PASS2 {i:,}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "user_id",
            "Ru", "label",
            "rating_entropy",
            "extremity_ratio",
            "average_rating_deviation",
            "review_burst_count",
            "lexical_diversity",
            "review_length_discrepancy",
        ])

        for uid, n in user_n.items():
            H = entropy(user_r[uid])
            ER = user_extreme[uid] / n if n else 0.0
            AAD = user_aad_sum[uid] / n if n else 0.0

            BC = 0
            for _, c in user_bucket_cnt[uid].items():
                if c > 1:
                    BC += (c - 1)

            LD = user_ttr_sum[uid] / n if n else 0.0
            RD = user_rd_sum[uid] / n if n else 0.0

            w.writerow([uid, user_Ru.get(uid), user_label.get(uid), H, ER, AAD, BC, LD, RD])

    print(f"[FEAT] Saved: {out_csv} | users={len(user_n):,}")

# =========================
# STEP 4: Merge features into JSONL
# =========================
def step4_merge_features(features_csv: Path, labeled_jsonl: Path, out_jsonl: Path):
    if SKIP_IF_EXISTS and out_jsonl.exists():
        print(f"[SKIP] labeled+features JSONL exists: {out_jsonl}")
        return

    assert_exists(features_csv, "FEATURES_CSV")
    assert_exists(labeled_jsonl, "LABELED_JSONL")

    user2feat = {}
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user2feat[row["user_id"]] = row

    print(f"[FEAT-MERGE] Loaded users: {len(user2feat):,}")

    missing = 0
    with open(labeled_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            r = json.loads(line)
            uid = r.get("user_id")
            row = user2feat.get(uid)

            if row is None:
                missing += 1
                for k in USER_FEATURE_KEYS[1:]:
                    r[k] = None
            else:
                r["rating_entropy"] = float(row["rating_entropy"]) if row["rating_entropy"] else None
                r["extremity_ratio"] = float(row["extremity_ratio"]) if row["extremity_ratio"] else None
                r["average_rating_deviation"] = float(row["average_rating_deviation"]) if row["average_rating_deviation"] else None
                r["review_burst_count"] = int(float(row["review_burst_count"])) if row["review_burst_count"] else None
                r["lexical_diversity"] = float(row["lexical_diversity"]) if row["lexical_diversity"] else None
                r["review_length_discrepancy"] = float(row["review_length_discrepancy"]) if row["review_length_discrepancy"] else None

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

            if i % PRINT_EVERY == 0:
                print(f"[FEAT-MERGE] {i:,} | missing users: {missing:,}")

    print(f"[FEAT-MERGE] Saved: {out_jsonl} | missing rows={missing:,}")

# =========================
# STEP 5: Graph build (PyG HeteroData) with parent_asin
# =========================
def pass1_build_maps_and_stats(jsonl_path: Path):
    user2idx = {}
    item2idx = {}

    user_feat_rows = []
    user_y = []

    item_sum = []
    item_cnt = []

    ts_min, ts_max = None, None
    E = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)
            uid = r.get("user_id")
            item_id = r.get(ITEM_ID_KEY)
            rating = r.get("rating")
            if uid is None or item_id is None or rating is None:
                continue

            uidx = user2idx.get(uid)
            if uidx is None:
                uidx = len(user2idx)
                user2idx[uid] = uidx
                user_feat_rows.append([np.nan] * len(USER_FEATURE_KEYS))
                lab = r.get("label", "unlabeled")
                user_y.append(LABEL_TO_INT.get(lab, -1))

            row = user_feat_rows[uidx]
            for j, k in enumerate(USER_FEATURE_KEYS):
                if math.isnan(row[j]):
                    row[j] = safe_float(r.get(k))

            iidx = item2idx.get(item_id)
            if iidx is None:
                iidx = len(item2idx)
                item2idx[item_id] = iidx
                item_sum.append(0.0)
                item_cnt.append(0)

            r_ui = safe_float(rating)
            if not math.isnan(r_ui):
                item_sum[iidx] += r_ui
                item_cnt[iidx] += 1

            ts = safe_int(r.get("timestamp"))
            if ts is not None:
                ts_min = ts if ts_min is None else min(ts_min, ts)
                ts_max = ts if ts_max is None else max(ts_max, ts)

            E += 1
            if i % PRINT_EVERY == 0:
                print(f"[GRAPH] PASS1 {i:,} | users={len(user2idx):,} items={len(item2idx):,} edges={E:,}")

    user_x = np.asarray(user_feat_rows, dtype=F_DTYPE)
    user_y = np.asarray(user_y, dtype=np.int64)

    item_sum = np.asarray(item_sum, dtype=np.float64)
    item_cnt = np.asarray(item_cnt, dtype=np.int64)

    item_mean = (item_sum / np.maximum(item_cnt, 1)).astype(F_DTYPE)
    item_x = np.stack([item_mean, item_cnt.astype(F_DTYPE)], axis=1).astype(F_DTYPE)

    stats = {
        "E": int(E),
        "ts_min": ts_min,
        "ts_max": ts_max,
        "num_users": int(user_x.shape[0]),
        "num_items": int(item_x.shape[0]),
        "item_id_key": ITEM_ID_KEY,
    }

    print("\n[GRAPH] PASS1 done:", stats)
    return user2idx, item2idx, user_x, user_y, item_mean, item_x, stats


def pass2_write_edges(jsonl_path: Path, user2idx, item2idx, item_mean, stats):
    ensure_outdir()

    E = stats["E"]
    ts_min, ts_max = stats["ts_min"], stats["ts_max"]

    src_path = OUT_DIR / "u2i_src.mmap"
    dst_path = OUT_DIR / "u2i_dst.mmap"
    attr_path = OUT_DIR / "u2i_attr.mmap"

    src = np.memmap(src_path, dtype=IDX_DTYPE, mode="w+", shape=(E,))
    dst = np.memmap(dst_path, dtype=IDX_DTYPE, mode="w+", shape=(E,))
    attr = np.memmap(attr_path, dtype=F_DTYPE, mode="w+", shape=(E, len(EDGE_ATTR_KEYS)))

    def norm_ts(ts):
        if not NORMALIZE_TIMESTAMP:
            return safe_float(ts)
        if ts_min is None or ts_max is None or ts_max == ts_min:
            return np.nan
        return (ts - ts_min) / (ts_max - ts_min)

    e = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            r = json.loads(line)

            uid = r.get("user_id")
            item_id = r.get(ITEM_ID_KEY)
            rating = r.get("rating")
            if uid is None or item_id is None or rating is None:
                continue

            uidx = user2idx.get(uid)
            iidx = item2idx.get(item_id)
            if uidx is None or iidx is None:
                continue

            src[e] = uidx
            dst[e] = iidx

            r_ui = safe_float(rating)
            rbar_i = float(item_mean[iidx]) if iidx < len(item_mean) else np.nan
            verified = 1.0 if bool(r.get("verified_purchase", False)) else 0.0
            align = calc_rating_align(r_ui, rbar_i)
            ts = safe_int(r.get("timestamp"))
            tsn = norm_ts(ts) if ts is not None else np.nan
            hv = safe_float(r.get("helpful_vote"))

            values = {
                "verified": verified,
                "rating_align": align,
                "rating": r_ui,
                "timestamp_norm": tsn,
                "helpful_vote": hv,
            }
            for j, k in enumerate(EDGE_ATTR_KEYS):
                attr[e, j] = safe_float(values.get(k))

            e += 1
            if i % PRINT_EVERY == 0:
                print(f"[GRAPH] PASS2 {i:,} | edges_written={e:,}/{E:,}")

    src.flush()
    dst.flush()
    attr.flush()
    print("\n[GRAPH] PASS2 done. memmaps saved in:", OUT_DIR)
    if e != E:
        print(f"[WARN] Expected edges={E:,} but wrote {e:,}.")

    return src_path, dst_path, attr_path


def save_mappings(user2idx, item2idx):
    with open(OUT_DIR / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_DIR / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[GRAPH] Saved mappings pkl files.")


def export_pyg(user_x, user_y, item_x, src_path: Path, dst_path: Path, attr_path: Path, stats):
    import torch
    from torch_geometric.data import HeteroData

    E = stats["E"]
    src = np.memmap(src_path, dtype=IDX_DTYPE, mode="r", shape=(E,))
    dst = np.memmap(dst_path, dtype=IDX_DTYPE, mode="r", shape=(E,))
    edge_attr = np.memmap(attr_path, dtype=F_DTYPE, mode="r", shape=(E, len(EDGE_ATTR_KEYS)))

    data = HeteroData()
    data["user"].x = torch.from_numpy(user_x.astype(np.float32))
    data["user"].y = torch.from_numpy(user_y.astype(np.int64))
    data["item"].x = torch.from_numpy(item_x.astype(np.float32))

    u2i_edge_index = np.vstack([src.astype(np.int64), dst.astype(np.int64)])
    data[("user", "rates", "item")].edge_index = torch.from_numpy(u2i_edge_index)
    data[("user", "rates", "item")].edge_attr = torch.from_numpy(edge_attr.astype(np.float32))

    i2u_edge_index = np.vstack([dst.astype(np.int64), src.astype(np.int64)])
    data[("item", "rev_rates", "user")].edge_index = torch.from_numpy(i2u_edge_index)
    data[("item", "rev_rates", "user")].edge_attr = data[("user", "rates", "item")].edge_attr

    out_pt = OUT_DIR / "graph_hetero_rates.pt"
    torch.save(data, out_pt)
    print("[GRAPH] Saved graph:", out_pt)
    return out_pt

def train_and_export_credibility(graph_pt: Path):
    """
    Implements thesis version:
    - SLAS sampling: p(u|i) ∝ exp(κ * sim_{u→i}) with sim_{u→i} = avg_{i'∈I(u)} S_{ii'}  (Eq. 3.10-3.11)
    - EWA weights: w_{u,i} = β*Verified + γ*RatingAlign                                 (Eq. 3.12)
    - GraphSAGE-style two-stage aggregation                                               (Eq. 3.13-3.16)
    - Losses: BCE + Smoothness + Temporal Contrastive, combined objective                (Eq. 3.18-3.21)
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # -----------------
    # Hyperparams (edit)
    # -----------------
    device = "cuda" if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[CRED] device = {device}")

    # EWA coefficients (Eq. 3.12)
    BETA = 1.0
    GAMMA = 1.0

    # SLAS (Eq. 3.10-3.11)
    SLAS_KAPPA = 3.0                 # κ
    SLAS_UPWEIGHT_LABELED = 1.0      # labeled up-weight factor (thesis says "up-weighted" but not exact value)
    K_USER_NEIGH = 15                # |S(i)| number of users sampled for each item
    K_ITEM_NEIGH = 15                # |S(u)| number of items sampled for each user

    # Loss weights (Eq. 3.21)
    LAMBDA_SMOOTH = 0.1              # λ_smooth
    LAMBDA_CONT = 0.1                # λ_cont

    # Temporal contrastive (Eq. 3.20)
    TAU_TEMP = 0.2                   # temperature τ
    # Temporal view split: use timestamp_norm (<split for view1, >=split for view2)
    TEMP_SPLIT = 0.5                 # you can randomize per step too

    # Smoothness: use only edges with positive normalized weight
    SMOOTH_MIN_W = 0.0

    # -----------------
    # Utils
    # -----------------
    def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
        # z1,z2: [B, d] positives are diagonal pairs
        z1 = l2_normalize(z1)
        z2 = l2_normalize(z2)
        logits = (z1 @ z2.t()) / tau                  # [B,B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    # -----------------
    # Model (GraphSAGE-style + EWA)
    # -----------------
    class CredModel(nn.Module):
        def __init__(self, user_in_dim, item_in_dim, hidden_dim, edge_attr_keys):
            super().__init__()
            self.user_proj = nn.Linear(user_in_dim, hidden_dim)
            self.item_proj = nn.Linear(item_in_dim, hidden_dim)

            self.item_upd = nn.Linear(hidden_dim * 2, hidden_dim)   # Eq 3.14
            self.user_upd = nn.Linear(hidden_dim * 2, hidden_dim)   # Eq 3.16
            self.out = nn.Linear(hidden_dim, 1)                     # Eq 3.17

            self.edge_attr_keys = edge_attr_keys
            self.EDGE_VERIFIED = edge_attr_keys.index("verified")
            self.EDGE_ALIGN = edge_attr_keys.index("rating_align")

        def ewa_raw(self, edge_attr: torch.Tensor) -> torch.Tensor:
            # Eq. (3.12): w_{u,i} = β*Verified + γ*RatingAlign
            verified = edge_attr[:, self.EDGE_VERIFIED].clamp(0, 1)
            align = edge_attr[:, self.EDGE_ALIGN]
            w = BETA * verified + GAMMA * align
            return w.clamp(min=0.0)

        def normalize_per_dst(self, w: torch.Tensor, dst: torch.Tensor, num_dst: int) -> torch.Tensor:
            denom = scatter_add(w.unsqueeze(-1), dst, dim_size=num_dst).squeeze(-1) + 1e-12
            return w / denom[dst]

        def aggregate(self, src_x: torch.Tensor, edge_index: torch.Tensor, w_tilde: torch.Tensor, num_dst: int) -> torch.Tensor:
            # edge_index: [2, E] with src=edge_index[0], dst=edge_index[1]
            src = edge_index[0]
            dst = edge_index[1]
            msg = w_tilde.unsqueeze(-1) * src_x[src]
            return scatter_add(msg, dst, dim_size=num_dst)

        def forward_subgraph(self, x_u, x_i, e_u2i, ea_u2i, e_i2u, ea_i2u):
            """
            x_u: [Uloc, Fu], x_i: [Iloc, Fi]
            returns:
              cred: [Uloc]
              h_u2: [Uloc, d]  (Eq 3.16 output)
              h_i1: [Iloc, d]  (Eq 3.14 output)
              w1t_for_edges: [E_u2i] normalized weights (useful for smoothness)
            """
            h_u0 = self.user_proj(x_u)   # Eq 3.9
            h_i0 = self.item_proj(x_i)   # Eq 3.9

            # ---- Layer 1: user -> item (Eq 3.13-3.14)
            w1 = self.ewa_raw(ea_u2i)
            dst_i = e_u2i[1]
            w1t = self.normalize_per_dst(w1, dst_i, num_dst=h_i0.size(0))
            m_i1 = self.aggregate(h_u0, e_u2i, w1t, num_dst=h_i0.size(0))
            h_i1 = F.relu(self.item_upd(torch.cat([h_i0, m_i1], dim=-1)))

            # ---- Layer 2: item -> user (Eq 3.15-3.16)
            w2 = self.ewa_raw(ea_i2u)
            dst_u = e_i2u[1]
            w2t = self.normalize_per_dst(w2, dst_u, num_dst=h_u0.size(0))
            m_u2 = self.aggregate(h_i1, e_i2u, w2t, num_dst=h_u0.size(0))
            h_u2 = F.relu(self.user_upd(torch.cat([h_u0, m_u2], dim=-1)))

            cred = torch.sigmoid(self.out(h_u2)).squeeze(-1)  # Eq 3.17
            return cred, h_u2, h_i1, w1t

    # -----------------
    # Load graph
    # -----------------
    data = torch.load(graph_pt, map_location="cpu", weights_only=False)

    # Global tensors
    user_x_all = data["user"].x.float()
    user_y_all = data["user"].y.long()
    item_x_all = data["item"].x.float()

    e_u2i_all = data[("user", "rates", "item")].edge_index.long()      # [2, E]
    ea_u2i_all = data[("user", "rates", "item")].edge_attr.float()     # [E, A]

    # (reverse edges share same attr in your export)
    e_i2u_all = data[("item", "rev_rates", "user")].edge_index.long()
    ea_i2u_all = data[("item", "rev_rates", "user")].edge_attr.float()

    U = user_x_all.size(0)
    I = item_x_all.size(0)
    E = e_u2i_all.size(1)

    # Indices used in sampling
    ts_idx = EDGE_ATTR_KEYS.index("timestamp_norm")

    # -----------------
    # Precompute item similarity proxies for SLAS (cosine on item features)
    # S_{ii'} is not specified in thesis; this is a reasonable default.
    # -----------------
    item_feat = item_x_all.clone()
    item_feat_norm = l2_normalize(item_feat)  # [I, Fi]

    # Compute per-user mean item feature vector (approx for avg_{i'∈I(u)} S_{ii'})
    # user_mu[u] ~ mean of item_feat_norm over items in I(u)
    src_u = e_u2i_all[0]  # [E]
    dst_i = e_u2i_all[1]  # [E]
    user_sum = torch.zeros((U, item_feat_norm.size(1)), dtype=torch.float32)
    user_sum.index_add_(0, src_u, item_feat_norm[dst_i])
    user_deg = torch.zeros((U,), dtype=torch.float32)
    user_deg.index_add_(0, src_u, torch.ones((E,), dtype=torch.float32))
    user_mu = user_sum / (user_deg.unsqueeze(-1).clamp(min=1.0))
    user_mu = l2_normalize(user_mu)  # [U, Fi]

    # -----------------
    # Build adjacency lists (CSR) for fast neighbor lookup
    # (Uses argsort; avoid Python loops over edges)
    # -----------------
    def build_csr_from_src(src: torch.Tensor, dst: torch.Tensor, num_src: int):
        src_np = src.numpy()
        dst_np = dst.numpy()
        order = np.argsort(src_np, kind="mergesort")  # stable, fast in C
        src_sorted = src_np[order]
        dst_sorted = dst_np[order]
        counts = np.bincount(src_sorted, minlength=num_src)
        ptr = np.zeros((num_src + 1,), dtype=np.int64)
        ptr[1:] = np.cumsum(counts, dtype=np.int64)
        eid_sorted = order.astype(np.int64)
        return ptr, dst_sorted.astype(np.int64), eid_sorted

    print("[CRED] Building CSR adjacencies...")
    u_ptr, u_nbr_items, u_eids = build_csr_from_src(e_u2i_all[0], e_u2i_all[1], U)  # user -> items
    i_ptr, i_nbr_users, i_eids = build_csr_from_src(e_u2i_all[1], e_u2i_all[0], I)  # item -> users
    print("[CRED] CSR built.")

    rng = np.random.default_rng(42)

    # -----------------
    # SLAS sampling
    # -----------------
    def slas_sample_items_for_user(u_global: int, k: int, temporal_view: str | None):
        """
        Returns sampled global item ids for user u (from its neighborhood).
        temporal_view: None | "early" | "late"
        """
        start, end = u_ptr[u_global], u_ptr[u_global + 1]
        if end <= start:
            return np.empty((0,), dtype=np.int64)

        items = u_nbr_items[start:end]        # global item ids
        eids = u_eids[start:end]              # global edge ids

        if temporal_view is not None:
            ts = ea_u2i_all[eids, ts_idx].numpy()
            if temporal_view == "early":
                mask = ts < TEMP_SPLIT
            else:
                mask = ts >= TEMP_SPLIT
            items = items[mask]
            eids = eids[mask]
            if items.size == 0:
                return np.empty((0,), dtype=np.int64)

        if items.size <= k:
            return items.copy()

        # sim_{u→i} ≈ cosine(item_i, mean_items_of_user_u)  (proxy for Eq 3.11)
        mu_u = user_mu[u_global]  # [Fi]
        sim = (item_feat_norm[torch.from_numpy(items)] @ mu_u).numpy()  # [deg]
        w = np.exp(SLAS_KAPPA * sim)
        w = w / (w.sum() + 1e-12)

        choice = rng.choice(items.size, size=k, replace=False, p=w)
        return items[choice]

    def slas_sample_users_for_item(i_global: int, k: int):
        """
        Returns sampled global user ids for item i (from its neighborhood).
        Implements p(u|i) ∝ exp(κ * sim_{u→i}) with labeled up-weight (Eq 3.10 + text).
        """
        start, end = i_ptr[i_global], i_ptr[i_global + 1]
        if end <= start:
            return np.empty((0,), dtype=np.int64)

        users = i_nbr_users[start:end]  # global user ids
        if users.size <= k:
            return users.copy()

        # similarity proxy: cosine(item_i, user_mu[u])
        v_i = item_feat_norm[i_global]  # [Fi]
        sim = (user_mu[torch.from_numpy(users)] @ v_i).numpy()

        w = np.exp(SLAS_KAPPA * sim)

        # labeled up-weighting (thesis: "Labeled nodes are up-weighted")
        y = user_y_all[torch.from_numpy(users)].numpy()
        labeled_mask = (y >= 0)
        w[labeled_mask] *= (1.0 + SLAS_UPWEIGHT_LABELED)

        w = w / (w.sum() + 1e-12)
        choice = rng.choice(users.size, size=k, replace=False, p=w)
        return users[choice]

    # -----------------
    # Subgraph builder for a seed user batch
    # -----------------
    def build_slas_subgraph(seed_users: np.ndarray, temporal_view: str | None):
        """
        Returns a sampled bipartite subgraph induced by:
        - seed_users
        - items sampled for each seed user (S(u))
        - users sampled for each sampled item (S(i))
        Local node ordering:
        - users_local[0:bs] are the seed users (for easy supervised loss)
        """
        bs = seed_users.size

        # 1) sample items for seeds
        sampled_items_list = []
        for u in seed_users:
            sampled_items_list.append(slas_sample_items_for_user(int(u), K_ITEM_NEIGH, temporal_view))
        sampled_items = np.unique(np.concatenate(sampled_items_list) if sampled_items_list else np.empty((0,), np.int64))

        # 2) sample extra users for sampled items
        extra_users_list = []
        for i_g in sampled_items:
            extra_users_list.append(slas_sample_users_for_item(int(i_g), K_USER_NEIGH))
        extra_users = np.unique(np.concatenate(extra_users_list) if extra_users_list else np.empty((0,), np.int64))

        # ensure seeds are included and first
        seed_set = set(seed_users.tolist())
        extra_only = np.array([u for u in extra_users.tolist() if u not in seed_set], dtype=np.int64)
        users_global = np.concatenate([seed_users, extra_only], axis=0)

        # local id maps
        user_gid2lid = {int(g): idx for idx, g in enumerate(users_global.tolist())}
        item_gid2lid = {int(g): idx for idx, g in enumerate(sampled_items.tolist())}

        # 3) collect edges u->i that stay inside sampled items
        src_l = []
        dst_l = []
        eid_l = []

        sampled_item_set = set(sampled_items.tolist())

        for ug in users_global:
            ug = int(ug)
            ustart, uend = u_ptr[ug], u_ptr[ug + 1]
            if uend <= ustart:
                continue
            items = u_nbr_items[ustart:uend]
            eids = u_eids[ustart:uend]

            if temporal_view is not None:
                ts = ea_u2i_all[eids, ts_idx].numpy()
                if temporal_view == "early":
                    mask = ts < TEMP_SPLIT
                else:
                    mask = ts >= TEMP_SPLIT
                items = items[mask]
                eids = eids[mask]

            for it, eid in zip(items.tolist(), eids.tolist()):
                if it in sampled_item_set:
                    src_l.append(user_gid2lid[ug])
                    dst_l.append(item_gid2lid[it])
                    eid_l.append(eid)

        if len(eid_l) == 0:
            # empty subgraph safeguard
            e_u2i = torch.zeros((2, 0), dtype=torch.long)
            ea_u2i = torch.zeros((0, ea_u2i_all.size(1)), dtype=torch.float32)
        else:
            e_u2i = torch.tensor([src_l, dst_l], dtype=torch.long)
            ea_u2i = ea_u2i_all[torch.tensor(eid_l, dtype=torch.long)].clone()

        # reverse edges i->u are just swapped with same attrs
        e_i2u = torch.stack([e_u2i[1], e_u2i[0]], dim=0)
        ea_i2u = ea_u2i

        # node features/labels
        x_u = user_x_all[torch.from_numpy(users_global)].clone()
        y_u = user_y_all[torch.from_numpy(users_global)].clone()
        x_i = item_x_all[torch.from_numpy(sampled_items)].clone()

        return {
            "bs": bs,
            "users_global": users_global,
            "items_global": sampled_items,
            "x_u": x_u,
            "y_u": y_u,
            "x_i": x_i,
            "e_u2i": e_u2i,
            "ea_u2i": ea_u2i,
            "e_i2u": e_i2u,
            "ea_i2u": ea_i2u,
        }

    # -----------------
    # Train/test split on labeled users
    # -----------------
    labeled_users = (user_y_all >= 0).nonzero(as_tuple=False).view(-1).numpy()
    if labeled_users.size == 0:
        raise RuntimeError("No labeled users found (y>=0). Check Ru labeling output.")

    rng.shuffle(labeled_users)
    split = int(0.8 * labeled_users.size)
    train_users = labeled_users[:split]
    print(f"[CRED] labeled users={labeled_users.size:,} | train={train_users.size:,}")

    # -----------------
    # Train
    # -----------------
    model = CredModel(user_x_all.size(1), item_x_all.size(1), HIDDEN_DIM, EDGE_ATTR_KEYS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def smoothness_loss(h_u2: torch.Tensor, h_i1: torch.Tensor, e_u2i: torch.Tensor, w_u2i_norm: torch.Tensor):
        """
        Thesis Eq (3.19): encourages connected nodes to be close, weighted by w~.
        In bipartite setting, we apply it on (u,i) edges in the sampled subgraph.
        """
        if e_u2i.size(1) == 0:
            return torch.tensor(0.0, device=h_u2.device)
        src = e_u2i[0]
        dst = e_u2i[1]
        w = w_u2i_norm
        mask = w > SMOOTH_MIN_W
        if not mask.any():
            return torch.tensor(0.0, device=h_u2.device)
        diff = h_u2[src[mask]] - h_i1[dst[mask]]
        return (w[mask] * (diff.pow(2).sum(dim=-1))).mean()

    # minibatch iterator
    def batches(arr, bs):
        for j in range(0, len(arr), bs):
            yield arr[j:j + bs]

    for ep in range(1, EPOCHS + 1):
        rng.shuffle(train_users)
        model.train()
        total_loss = 0.0
        nsteps = 0

        for seed in batches(train_users, BATCH_SIZE):
            seed = np.asarray(seed, dtype=np.int64)

            # Two temporal perspectives for contrastive loss (Eq 3.20)
            g1 = build_slas_subgraph(seed, temporal_view="early")
            g2 = build_slas_subgraph(seed, temporal_view="late")

            # Move to device
            for g in (g1, g2):
                g["x_u"] = g["x_u"].to(device)
                g["y_u"] = g["y_u"].to(device)
                g["x_i"] = g["x_i"].to(device)
                g["e_u2i"] = g["e_u2i"].to(device)
                g["ea_u2i"] = g["ea_u2i"].to(device)
                g["e_i2u"] = g["e_i2u"].to(device)
                g["ea_i2u"] = g["ea_i2u"].to(device)

            opt.zero_grad()

            pred1, h_u2_1, h_i1_1, w1t_1 = model.forward_subgraph(
                g1["x_u"], g1["x_i"], g1["e_u2i"], g1["ea_u2i"], g1["e_i2u"], g1["ea_i2u"]
            )
            pred2, h_u2_2, h_i1_2, w1t_2 = model.forward_subgraph(
                g2["x_u"], g2["x_i"], g2["e_u2i"], g2["ea_u2i"], g2["e_i2u"], g2["ea_i2u"]
            )

            bs = g1["bs"]

            # ---- Supervised BCE (Eq 3.18)
            y = g1["y_u"][:bs]
            keep = y >= 0
            loss_sup = (
                F.binary_cross_entropy(pred1[:bs][keep], y[keep].float())
                if keep.any()
                else torch.tensor(0.0, device=device)
            )

            # ---- Smoothness (Eq 3.19) on view1 subgraph
            loss_smooth = smoothness_loss(h_u2_1, h_i1_1, g1["e_u2i"], w1t_1)

            # ---- Temporal contrastive (Eq 3.20): positives are same user across time views
            loss_cont = info_nce(h_u2_1[:bs], h_u2_2[:bs], tau=TAU_TEMP)

            # ---- Final objective (Eq 3.21)
            loss = loss_sup + LAMBDA_SMOOTH * loss_smooth + LAMBDA_CONT * loss_cont
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            nsteps += 1

        print(f"[CRED] Epoch {ep:02d} | loss={total_loss/max(nsteps,1):.4f}")

    # -----------------
    # Inference: credibility for all users (single view)
    # -----------------
    model.eval()
    cred_all = torch.empty((U,), dtype=torch.float32)

    with torch.no_grad():
        all_users = np.arange(U, dtype=np.int64)
        for seed in batches(all_users, BATCH_SIZE):
            seed = np.asarray(seed, dtype=np.int64)
            g = build_slas_subgraph(seed, temporal_view=None)

            # move
            x_u = g["x_u"].to(device)
            x_i = g["x_i"].to(device)
            e_u2i = g["e_u2i"].to(device)
            ea_u2i = g["ea_u2i"].to(device)
            e_i2u = g["e_i2u"].to(device)
            ea_i2u = g["ea_i2u"].to(device)

            pred, _, _, _ = model.forward_subgraph(x_u, x_i, e_u2i, ea_u2i, e_i2u, ea_i2u)
            bs = g["bs"]
            cred_all[torch.from_numpy(seed)] = pred[:bs].detach().cpu()

    # -----------------
    # Save outputs
    # -----------------
    out_npy = OUT_DIR/ "/credibility_scores.npy"
    out_csv = OUT_DIR / "/credibility_scores.csv"
    out_pt = OUT_DIR / "/cred_model.pt"

    np.save(out_npy, cred_all.numpy())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("user_idx,credibility\n")
        for i, v in enumerate(cred_all.numpy()):
            f.write(f"{i},{float(v):.6f}\n")

    torch.save(model.state_dict(), out_pt)

    print(f"[CRED] Saved: {out_npy}")
    print(f"[CRED] Saved: {out_csv}")
    print(f"[CRED] Saved: {out_pt}")
