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

Outputs:
- dataset/user_labels.csv
- dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl
- dataset/user_features.csv
- dataset/Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl
- dataset/graph_pyg_parent_asin/graph_hetero_rates.pt
- dataset/graph_pyg_parent_asin/credibility_scores.npy + credibility_scores.csv + cred_model.pt

Dependencies:
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
# CONFIG (EDIT THESE)
# =========================
RAW_JSONL = "dataset/Clothing_Shoes_and_Jewelry.jsonl"

LABELS_CSV = "dataset/user_labels.csv"
LABELED_JSONL = "dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl"

FEATURES_CSV = "dataset/user_features.csv"
LABELED_FEATURED_JSONL = "dataset/Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl"

OUT_DIR = Path("dataset/graph_pyg_parent_asin")

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
EPOCHS = 3
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


# =========================
# STEP 1: User labeling (Ru)
# =========================
def step1_build_user_labels(raw_jsonl: str, out_csv: str):
    if SKIP_IF_EXISTS and Path(out_csv).exists():
        print(f"[SKIP] labels CSV exists: {out_csv}")
        return

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
def step2_merge_labels(raw_jsonl: str, labels_csv: str, out_jsonl: str):
    if SKIP_IF_EXISTS and Path(out_jsonl).exists():
        print(f"[SKIP] labeled JSONL exists: {out_jsonl}")
        return

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
def step3_compute_user_features(labeled_jsonl: str, out_csv: str):
    if SKIP_IF_EXISTS and Path(out_csv).exists():
        print(f"[SKIP] features CSV exists: {out_csv}")
        return

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
            item_id = r.get(ITEM_ID_KEY)      # ✅ parent_asin
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
            item_id = r.get(ITEM_ID_KEY)      # ✅ parent_asin
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

    # SAVE
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
def step4_merge_features(features_csv: str, labeled_jsonl: str, out_jsonl: str):
    if SKIP_IF_EXISTS and Path(out_jsonl).exists():
        print(f"[SKIP] labeled+features JSONL exists: {out_jsonl}")
        return

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
def pass1_build_maps_and_stats(jsonl_path: str):
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
            item_id = r.get(ITEM_ID_KEY)  # ✅ parent_asin
            rating = r.get("rating")
            if uid is None or item_id is None or rating is None:
                continue

            # user map
            uidx = user2idx.get(uid)
            if uidx is None:
                uidx = len(user2idx)
                user2idx[uid] = uidx
                user_feat_rows.append([np.nan] * len(USER_FEATURE_KEYS))
                lab = r.get("label", "unlabeled")
                user_y.append(LABEL_TO_INT.get(lab, -1))

            # fill features once
            row = user_feat_rows[uidx]
            for j, k in enumerate(USER_FEATURE_KEYS):
                if math.isnan(row[j]):
                    row[j] = safe_float(r.get(k))

            # item map
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


def pass2_write_edges(jsonl_path: str, user2idx, item2idx, item_mean, stats):
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
            item_id = r.get(ITEM_ID_KEY)  # ✅ parent_asin
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

    src.flush(); dst.flush(); attr.flush()
    print("\n[GRAPH] PASS2 done. memmaps saved in:", OUT_DIR)
    if e != E:
        print(f"[WARN] Expected edges={E:,} but wrote {e:,}.")

    return str(src_path), str(dst_path), str(attr_path)


def save_mappings(user2idx, item2idx):
    with open(OUT_DIR / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_DIR / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[GRAPH] Saved mappings pkl files.")


def export_pyg(user_x, user_y, item_x, src_path, dst_path, attr_path, stats):
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


# =========================
# STEP 6: Credibility model (GraphSAGE-style + EWA)
# =========================
def train_and_export_credibility(graph_pt: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import NeighborLoader

    device = "cuda" if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[CRED] device = {device}")

    def scatter_add(src, index, dim_size):
        out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    class CredModel(nn.Module):
        def __init__(self, user_in_dim, item_in_dim, hidden_dim):
            super().__init__()
            self.user_proj = nn.Linear(user_in_dim, hidden_dim)
            self.item_proj = nn.Linear(item_in_dim, hidden_dim)
            self.item_upd = nn.Linear(hidden_dim * 2, hidden_dim)
            self.user_upd = nn.Linear(hidden_dim * 2, hidden_dim)
            self.out = nn.Linear(hidden_dim, 1)

            self.EDGE_VERIFIED = EDGE_ATTR_KEYS.index("verified")
            self.EDGE_ALIGN = EDGE_ATTR_KEYS.index("rating_align")

        def _ewa(self, edge_attr):
            verified = edge_attr[:, self.EDGE_VERIFIED].clamp(0, 1)
            align = edge_attr[:, self.EDGE_ALIGN]
            w = verified + align
            return w.clamp(min=0.0)

        def _normalize_per_dst(self, w, dst, num_dst):
            denom = scatter_add(w.unsqueeze(-1), dst, dim_size=num_dst).squeeze(-1) + 1e-12
            return w / denom[dst]

        def _aggregate(self, src_x, edge_index, w_tilde, num_dst):
            src = edge_index[0]
            dst = edge_index[1]
            msg = w_tilde.unsqueeze(-1) * src_x[src]
            return scatter_add(msg, dst, dim_size=num_dst)

        def forward(self, batch):
            x_u = batch["user"].x
            x_i = batch["item"].x
            h_u0 = self.user_proj(x_u)
            h_i0 = self.item_proj(x_i)

            e_u2i = batch[("user", "rates", "item")]
            e_i2u = batch[("item", "rev_rates", "user")]

            # user -> item
            w1 = self._ewa(e_u2i.edge_attr)
            dst_i = e_u2i.edge_index[1]
            w1t = self._normalize_per_dst(w1, dst_i, num_dst=h_i0.size(0))
            m_i = self._aggregate(h_u0, e_u2i.edge_index, w1t, num_dst=h_i0.size(0))
            h_i1 = F.relu(self.item_upd(torch.cat([h_i0, m_i], dim=-1)))

            # item -> user
            w2 = self._ewa(e_i2u.edge_attr)
            dst_u = e_i2u.edge_index[1]
            w2t = self._normalize_per_dst(w2, dst_u, num_dst=h_u0.size(0))
            m_u = self._aggregate(h_i1, e_i2u.edge_index, w2t, num_dst=h_u0.size(0))
            h_u1 = F.relu(self.user_upd(torch.cat([h_u0, m_u], dim=-1)))

            cred = torch.sigmoid(self.out(h_u1)).squeeze(-1)
            return cred

    def make_train_mask(y_all, seed=42):
        labeled = (y_all >= 0).nonzero(as_tuple=False).view(-1)
        if labeled.numel() == 0:
            raise RuntimeError("No labeled users found (y>=0). Check Ru labeling output.")
        g = torch.Generator().manual_seed(seed)
        perm = labeled[torch.randperm(labeled.numel(), generator=g)]
        train_idx = perm[: int(0.8 * perm.numel())]
        mask = torch.zeros_like(y_all, dtype=torch.bool)
        mask[train_idx] = True
        return mask

    data = torch.load(graph_pt, map_location="cpu")
    train_mask = make_train_mask(data["user"].y)

    num_neighbors = {
        ("user", "rates", "item"): [15, 0],
        ("item", "rev_rates", "user"): [0, 15],
    }

    train_seed = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborLoader(
        data,
        input_nodes=("user", train_seed),
        num_neighbors=num_neighbors,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = CredModel(data["user"].x.size(-1), data["item"].x.size(-1), HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for ep in range(1, EPOCHS + 1):
        tot, nb = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)

            bs = batch["user"].batch_size
            y = batch["user"].y[:bs]
            keep = y >= 0
            loss = (
                torch.nn.functional.binary_cross_entropy(pred[:bs][keep], y[keep].float())
                if keep.any()
                else torch.tensor(0.0, device=device)
            )
            loss.backward()
            opt.step()

            tot += float(loss.detach().cpu())
            nb += 1
        print(f"[CRED] Epoch {ep:02d} | loss={tot/max(nb,1):.4f}")

    # Inference for all users
    all_loader = NeighborLoader(
        data,
        input_nodes=("user", torch.arange(data["user"].num_nodes)),
        num_neighbors=num_neighbors,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    cred = torch.empty((data["user"].num_nodes,), dtype=torch.float32)
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            pred = model(batch)
            bs = batch["user"].batch_size
            idx = batch["user"].n_id[:bs]
            cred[idx] = pred[:bs].detach().cpu()

    out_npy = OUT_DIR / "credibility_scores.npy"
    out_csv = OUT_DIR / "credibility_scores.csv"
    out_pt = OUT_DIR / "cred_model.pt"

    np.save(out_npy, cred.numpy())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("user_idx,credibility\n")
        for i, v in enumerate(cred.numpy()):
            f.write(f"{i},{float(v):.6f}\n")

    import torch
    torch.save(model.state_dict(), out_pt)

    print(f"[CRED] Saved: {out_npy}")
    print(f"[CRED] Saved: {out_csv}")
    print(f"[CRED] Saved: {out_pt}")


# =========================
# MAIN
# =========================
def main():
    ensure_outdir()

    step1_build_user_labels(RAW_JSONL, LABELS_CSV)
    step2_merge_labels(RAW_JSONL, LABELS_CSV, LABELED_JSONL)
    step3_compute_user_features(LABELED_JSONL, FEATURES_CSV)
    step4_merge_features(FEATURES_CSV, LABELED_JSONL, LABELED_FEATURED_JSONL)

    print("\n=== GRAPH BUILD (item = parent_asin) ===")
    user2idx, item2idx, user_x, user_y, item_mean, item_x, stats = pass1_build_maps_and_stats(LABELED_FEATURED_JSONL)

    np.save(OUT_DIR / "user_x.npy", user_x)
    np.save(OUT_DIR / "user_y.npy", user_y)
    np.save(OUT_DIR / "item_x.npy", item_x)
    np.save(OUT_DIR / "item_mean.npy", item_mean)
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    save_mappings(user2idx, item2idx)
    src_path, dst_path, attr_path = pass2_write_edges(LABELED_FEATURED_JSONL, user2idx, item2idx, item_mean, stats)
    graph_pt = export_pyg(user_x, user_y, item_x, src_path, dst_path, attr_path, stats)

    print("\n=== CREDIBILITY MODEL ===")
    train_and_export_credibility(graph_pt)

    print("\nDONE. Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
