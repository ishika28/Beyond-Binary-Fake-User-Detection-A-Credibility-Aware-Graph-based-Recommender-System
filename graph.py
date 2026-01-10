# Auto-generated from index.ipynb
# Markdown cells are converted to comments.

# --- Code cell 1 ---
import pandas as pd
import numpy as np

# --- Code cell 2 ---
from collections import defaultdict

# --- Code cell 3 ---
import json
import math
import pickle
from pathlib import Path

path = "dataset/Clothing_Shoes_and_Jewelry.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)   # one JSON object

        # just to confirm it works
        if i <= 3:
            print(record)

        if i % 1_000_000 == 0:
            print(f"Loaded {i:,} rows")

# --- Code cell 4 ---
import csv

# --- Code cell 5 ---
out_csv = "dataset/user_labels.csv"

total_reviews = defaultdict(int)
helpful_reviews = defaultdict(int)

# 1) PASS: build user counts
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)
        u = r.get("user_id")
        if not u:
            continue

        total_reviews[u] += 1

        # thesis rule: helpful review if helpful_vote > 5
        if int(r.get("helpful_vote", 0)) > 5:
            helpful_reviews[u] += 1

        if i % 1_000_000 == 0:
            print(f"Processed {i:,} reviews")

# 2) Create labels + save
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "total_reviews", "helpful_reviews", "Ru", "label"])

    for u, tot in total_reviews.items():
        hel = helpful_reviews[u]
        Ru = hel / tot

        if Ru >= 0.7:
            label = "genuine"
        elif Ru <= 0.3:
            label = "fake"
        else:
            label = "unlabeled"

        writer.writerow([u, tot, hel, Ru, label])

print("Saved:", out_csv)
print("Total users labeled:", len(total_reviews))

# --- Code cell 6 ---


df_labels = pd.read_csv("dataset/user_labels.csv")
df_labels.head()

# --- Code cell 7 ---

labels_csv = "dataset/user_labels.csv"
input_jsonl = "dataset/Clothing_Shoes_and_Jewelry.jsonl"
output_jsonl = "dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl"

# 1) Load user labels into memory (dict)
user2lab = {}
with open(labels_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        u = row["user_id"]
        Ru = float(row["Ru"])
        label = row["label"]
        user2lab[u] = (Ru, label)

print("Loaded users from CSV:", len(user2lab))

# 2) Stream JSONL and write enriched JSONL
missing = 0
written = 0

with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        r = json.loads(line)

        u = r.get("user_id")
        if u in user2lab:
            Ru, label = user2lab[u]
            r["Ru"] = Ru
            r["label"] = label
        else:
            # If some user_id isn't present in CSV
            r["Ru"] = None
            r["label"] = None
            missing += 1

        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += 1

        if i % 1_000_000 == 0:
            print(f"Processed {i:,} rows | missing users: {missing:,}")

print("Done.")
print("Output:", output_jsonl)
print("Total rows written:", written)
print("Rows with missing label:", missing)

# --- Code cell 8 ---
label_path = "dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl"

with open(label_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)   # one JSON object

        # just to confirm it works
        if i <= 3:
            print(record)

        if i % 1_000_000 == 0:
            print(f"Loaded {i:,} rows")

# --- Code cell 9 ---
import json
import math
import re
import csv
from collections import defaultdict

# --- Code cell 10 ---
JSONL_PATH = "dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl"
OUT_CSV = "dataset/user_features.csv"

TAU_MS = 24 * 60 * 60 * 1000  # 1 day (adjust)

token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def tokenize(text: str):
    if not text:
        return []
    return token_re.findall(text.lower())

# -------------------------
# PASS 1: collect aggregates
# -------------------------

# user aggregates
user_n = defaultdict(int)
user_r = defaultdict(lambda: [0,0,0,0,0])          # rating counts 1..5
user_extreme = defaultdict(int)                    # count ratings in {1,5}
user_words_sum = defaultdict(int)                  # total words
user_ttr_sum = defaultdict(float)                  # sum of per-review TTR (proxy for LD)
user_bucket_cnt = defaultdict(lambda: defaultdict(int))  # (user -> bucket -> count) for burst approx

# Ru/label lookup (already inside JSONL, but we store one copy per user)
user_Ru = {}
user_label = {}

# item aggregates for item mean r̄_i (for AAD)
item_sum = defaultdict(float)
item_cnt = defaultdict(int)

# global mean review length ℓ̄
global_len_sum = 0
global_len_cnt = 0

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)

        uid = r.get("user_id")
        asin = r.get("asin")
        rating = r.get("rating")
        if not uid or not asin or rating is None:
            continue

        # store Ru/label once (they are already in your jsonl)
        if uid not in user_Ru:
            user_Ru[uid] = r.get("Ru", None)
            user_label[uid] = r.get("label", None)

        ri = int(round(float(rating)))
        ri = 1 if ri < 1 else 5 if ri > 5 else ri

        user_n[uid] += 1
        user_r[uid][ri-1] += 1
        if ri in (1, 5):
            user_extreme[uid] += 1

        # item mean stats (for AAD)
        item_sum[asin] += ri
        item_cnt[asin] += 1

        # text length + lexical proxy
        text = (r.get("title") or "") + " " + (r.get("text") or "")
        toks = tokenize(text)
        L = len(toks)

        user_words_sum[uid] += L
        global_len_sum += L
        global_len_cnt += 1

        # LD proxy = per-review type-token ratio
        if L > 0:
            user_ttr_sum[uid] += len(set(toks)) / L

        # Burst approx by τ-bucket
        ts = r.get("timestamp")
        if ts is not None:
            try:
                bucket = int(int(ts) // TAU_MS)
                user_bucket_cnt[uid][bucket] += 1
            except:
                pass

        if i % 1_000_000 == 0:
            print(f"Processed {i:,} reviews")

global_avg_len = global_len_sum / max(global_len_cnt, 1)

# compute item means
item_mean = {a: item_sum[a]/item_cnt[a] for a in item_cnt}
print("Item means computed:", len(item_mean))
print("Global avg length:", global_avg_len)

# -------------------------
# PASS 2: compute AAD & RD sums per user
# -------------------------
user_aad_sum = defaultdict(float)
user_rd_sum = defaultdict(float)

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)

        uid = r.get("user_id")
        asin = r.get("asin")
        rating = r.get("rating")
        if not uid or not asin or rating is None:
            continue

        ri = int(round(float(rating)))
        ri = 1 if ri < 1 else 5 if ri > 5 else ri

        rbar = item_mean.get(asin)
        if rbar is None:
            continue

        user_aad_sum[uid] += abs(ri - rbar)

        text = (r.get("title") or "") + " " + (r.get("text") or "")
        L = len(tokenize(text))
        user_rd_sum[uid] += abs(L - global_avg_len)

        if i % 1_000_000 == 0:
            print(f"PASS2 processed {i:,} reviews")

# -------------------------
# Final: compute 6 features + save
# -------------------------
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

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "user_id",
        "Ru", "label",
        "rating_entropy",               # (3.2)
        "extremity_ratio",              # (3.3)
        "average_rating_deviation",     # (3.4)
        "review_burst_count",           # (3.5) approx
        "lexical_diversity",            # (3.6) proxy
        "review_length_discrepancy"     # (3.7)
    ])

    for uid in user_n:
        n = user_n[uid]

        H = entropy(user_r[uid])
        ER = user_extreme[uid] / n if n else 0.0
        AAD = user_aad_sum[uid] / n if n else 0.0

        # burst approx: sum over buckets of (count-1)
        BC = 0
        for b, c in user_bucket_cnt[uid].items():
            if c > 1:
                BC += (c - 1)

        # LD proxy: mean per-review TTR
        LD = user_ttr_sum[uid] / n if n else 0.0

        RD = user_rd_sum[uid] / n if n else 0.0

        w.writerow([
            uid,
            user_Ru.get(uid),
            user_label.get(uid),
            H, ER, AAD, BC, LD, RD
        ])

print("Saved features to:", OUT_CSV)
print("Users:", len(user_n))

# --- Code cell 11 ---


path_features = "dataset/user_features.csv"
chunks = pd.read_csv(path_features, chunksize=500_000)   # adjust 200k–1M

for i, chunk in enumerate(chunks, 1):
    print(i, chunk.shape)
    # example: count labels per chunk
    print(chunk["label"].value_counts(dropna=False).head())
    if i == 3:
        break

# --- Code cell 12 ---
chunk.head()

# --- Code cell 13 ---


FEATURES_CSV = "dataset/user_features.csv"
IN_JSONL = "dataset/Clothing_Shoes_and_Jewelry_with_labels.jsonl"
OUT_JSONL = "dataset/Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl"

# 1) Load user features into memory (DICT)
user2feat = {}

with open(FEATURES_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        uid = row["user_id"]
        user2feat[uid] = (
            float(row["rating_entropy"]) if row["rating_entropy"] else None,
            float(row["extremity_ratio"]) if row["extremity_ratio"] else None,
            float(row["average_rating_deviation"]) if row["average_rating_deviation"] else None,
            int(float(row["review_burst_count"])) if row["review_burst_count"] else None,
            float(row["lexical_diversity"]) if row["lexical_diversity"] else None,
            float(row["review_length_discrepancy"]) if row["review_length_discrepancy"] else None,
        )

        if i % 1_000_000 == 0:
            print(f"Loaded {i:,} users into dict")

print("Total users in dict:", len(user2feat))

# 2) Stream JSONL and append features
missing = 0
with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        r = json.loads(line)
        uid = r.get("user_id")

        feats = user2feat.get(uid)
        if feats is None:
            missing += 1
            r["rating_entropy"] = None
            r["extremity_ratio"] = None
            r["average_rating_deviation"] = None
            r["review_burst_count"] = None
            r["lexical_diversity"] = None
            r["review_length_discrepancy"] = None
        else:
            H, ER, AAD, BC, LD, RD = feats
            r["rating_entropy"] = H
            r["extremity_ratio"] = ER
            r["average_rating_deviation"] = AAD
            r["review_burst_count"] = BC
            r["lexical_diversity"] = LD
            r["review_length_discrepancy"] = RD

        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        if i % 1_000_000 == 0:
            print(f"Processed {i:,} reviews | missing users: {missing:,}")

print("Saved:", OUT_JSONL)
print("Missing-feature rows:", missing)
JSONL_PATH = "dataset/Clothing_Shoes_and_Jewelry_with_labels_and_features.jsonl"
OUT_DIR = Path("dataset/graph_pyg_rates")

# 6 engineered features + Ru (already inside each JSONL row)
USER_FEATURE_KEYS = [
    "Ru",
    "rating_entropy",
    "extremity_ratio",
    "average_rating_deviation",
    "review_burst_count",
    "lexical_diversity",
    "review_length_discrepancy",
]

# User labels for GraphSAGE user classification
# -1 means unlabeled (mask in loss)
LABEL_TO_INT = {"fake": 0, "genuine": 1, "unlabeled": -1}

# Edge attributes (EWA-ready)
EDGE_ATTR_KEYS = ["verified", "rating_align", "rating", "timestamp_norm", "helpful_vote"]

# Timestamp normalization
NORMALIZE_TIMESTAMP = True

# Disk/RAM-friendly dtypes
IDX_DTYPE = np.int32
F_DTYPE = np.float32

PRINT_EVERY = 1_000_000

# =========================
# Helpers
# =========================
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
    # RatingAlign_{u,i} = 1 - |r_ui - rbar_i| / 4
    # assuming rating scale 1..5
    if r_ui is None or rbar_i is None or math.isnan(r_ui) or math.isnan(rbar_i):
        return np.nan
    return 1.0 - (abs(float(r_ui) - float(rbar_i)) / 4.0)

# =========================
# PASS 1
# Build:
# - user2idx, item2idx
# - user_x (Ru + 6 features)
# - user_y (fake/genuine, unlabeled=-1)
# - item_mean_rating, item_count -> item_x
# - timestamp min/max
# - edge_count E
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
            asin = r.get("asin")
            rating = r.get("rating")

            # For "rates" edge, rating must exist:
            if uid is None or asin is None or rating is None:
                continue

            # user mapping
            uidx = user2idx.get(uid)
            if uidx is None:
                uidx = len(user2idx)
                user2idx[uid] = uidx
                user_feat_rows.append([np.nan] * len(USER_FEATURE_KEYS))

                lab = r.get("label", "unlabeled")
                user_y.append(LABEL_TO_INT.get(lab, -1))

            # fill user features once
            row = user_feat_rows[uidx]
            for j, k in enumerate(USER_FEATURE_KEYS):
                if math.isnan(row[j]):
                    row[j] = safe_float(r.get(k))

            # item mapping
            iidx = item2idx.get(asin)
            if iidx is None:
                iidx = len(item2idx)
                item2idx[asin] = iidx
                item_sum.append(0.0)
                item_cnt.append(0)

            # item mean accumulators
            r_ui = safe_float(rating)
            if not math.isnan(r_ui):
                item_sum[iidx] += r_ui
                item_cnt[iidx] += 1

            # timestamp range
            ts = r.get("timestamp")
            if ts is not None:
                ts = safe_int(ts)
                if ts is not None:
                    ts_min = ts if ts_min is None else min(ts_min, ts)
                    ts_max = ts if ts_max is None else max(ts_max, ts)

            E += 1

            if i % PRINT_EVERY == 0:
                print(f"PASS1 {i:,} lines | users={len(user2idx):,} items={len(item2idx):,} edges={E:,}")

    user_x = np.asarray(user_feat_rows, dtype=F_DTYPE)
    user_y = np.asarray(user_y, dtype=np.int64)

    item_sum = np.asarray(item_sum, dtype=np.float64)
    item_cnt = np.asarray(item_cnt, dtype=np.int64)

    item_mean = (item_sum / np.maximum(item_cnt, 1)).astype(F_DTYPE)
    item_cnt_f = item_cnt.astype(F_DTYPE)

    # item node features: [mean_rating, rating_count]
    item_x = np.stack([item_mean, item_cnt_f], axis=1).astype(F_DTYPE)

    stats = {
        "E": int(E),
        "ts_min": ts_min,
        "ts_max": ts_max,
        "num_users": int(user_x.shape[0]),
        "num_items": int(item_x.shape[0]),
    }

    print("\nPASS1 done:", stats)
    return user2idx, item2idx, user_x, user_y, item_mean, item_x, stats

# =========================
# PASS 2
# Write edges to memmaps:
# - u2i_src, u2i_dst
# - u2i_attr: verified, rating_align, rating, timestamp_norm, helpful_vote
# =========================
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
            asin = r.get("asin")
            rating = r.get("rating")

            if uid is None or asin is None or rating is None:
                continue

            uidx = user2idx.get(uid)
            iidx = item2idx.get(asin)
            if uidx is None or iidx is None:
                continue

            src[e] = uidx
            dst[e] = iidx

            r_ui = safe_float(rating)
            rbar_i = float(item_mean[iidx]) if iidx < len(item_mean) else np.nan

            verified = 1.0 if bool(r.get("verified_purchase", False)) else 0.0
            align = calc_rating_align(r_ui, rbar_i)

            ts = r.get("timestamp")
            ts = safe_int(ts)
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
                print(f"PASS2 {i:,} lines | edges_written={e:,}/{E:,}")

    src.flush(); dst.flush(); attr.flush()
    print("\nPASS2 done. Edge memmaps saved in:", OUT_DIR)

    if e != E:
        print(f"[WARN] Expected edges={E:,} but wrote edges={e:,}. Some rows were skipped unexpectedly.")

    return str(src_path), str(dst_path), str(attr_path)

# =========================
# Save mappings (use pickle; JSON is huge)
# =========================
def save_mappings(user2idx, item2idx):
    with open(OUT_DIR / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_DIR / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved mappings:", OUT_DIR / "user2idx.pkl", "and", OUT_DIR / "item2idx.pkl")

# =========================
# Export PyG HeteroData
# =========================
def export_pyg(user_x, user_y, item_x, src_path, dst_path, attr_path, stats):
    import torch
    from torch_geometric.data import HeteroData

    E = stats["E"]

    src = np.memmap(src_path, dtype=IDX_DTYPE, mode="r", shape=(E,))
    dst = np.memmap(dst_path, dtype=IDX_DTYPE, mode="r", shape=(E,))
    edge_attr = np.memmap(attr_path, dtype=F_DTYPE, mode="r", shape=(E, len(EDGE_ATTR_KEYS)))

    data = HeteroData()

    # nodes
    data["user"].x = torch.from_numpy(user_x.astype(np.float32))
    data["user"].y = torch.from_numpy(user_y.astype(np.int64))  # -1 unlabeled

    data["item"].x = torch.from_numpy(item_x.astype(np.float32))

    # edges user->item
    u2i_edge_index = np.vstack([src.astype(np.int64), dst.astype(np.int64)])
    data[("user", "rates", "item")].edge_index = torch.from_numpy(u2i_edge_index)
    data[("user", "rates", "item")].edge_attr = torch.from_numpy(edge_attr.astype(np.float32))

    # reverse edges for two-stage propagation (user->item->user)
    i2u_edge_index = np.vstack([dst.astype(np.int64), src.astype(np.int64)])
    data[("item", "rev_rates", "user")].edge_index = torch.from_numpy(i2u_edge_index)
    data[("item", "rev_rates", "user")].edge_attr = data[("user", "rates", "item")].edge_attr

    out_pt = OUT_DIR / "graph_hetero_rates.pt"
    torch.save(data, out_pt)
    print("Saved PyG graph:", out_pt)

    meta = {
        "user_feature_keys": USER_FEATURE_KEYS,
        "edge_attr_keys": EDGE_ATTR_KEYS,
        "label_map": LABEL_TO_INT,
        "normalize_timestamp": NORMALIZE_TIMESTAMP,
        "num_users": stats["num_users"],
        "num_items": stats["num_items"],
        "E": stats["E"],
    }
    with open(OUT_DIR / "graph_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta:", OUT_DIR / "graph_meta.json")

# =========================
# MAIN
# =========================
def main():
    ensure_outdir()

    print("=== PASS 1/3: Build mappings + node features + item mean rating ===")
    user2idx, item2idx, user_x, user_y, item_mean, item_x, stats = pass1_build_maps_and_stats(JSONL_PATH)

    print("\n=== Save node arrays + stats ===")
    np.save(OUT_DIR / "user_x.npy", user_x)
    np.save(OUT_DIR / "user_y.npy", user_y)
    np.save(OUT_DIR / "item_x.npy", item_x)
    np.save(OUT_DIR / "item_mean.npy", item_mean)
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved arrays in:", OUT_DIR)

    print("\n=== Save ID mappings (pickle) ===")
    save_mappings(user2idx, item2idx)

    print("\n=== PASS 2/3: Write edges + edge attributes to memmaps ===")
    src_path, dst_path, attr_path = pass2_write_edges(JSONL_PATH, user2idx, item2idx, item_mean, stats)

    print("\n=== PASS 3/3: Export PyG HeteroData ===")
    export_pyg(user_x, user_y, item_x, src_path, dst_path, attr_path, stats)

    print("\nDONE. Output folder:", OUT_DIR)

if __name__ == "__main__":
    main()



