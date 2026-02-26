# raw_lightgcn_with_cred_message_passing.py
# Your raw LightGCN pipeline + credibility used INSIDE message passing (Eq 3.23/3.24 style).
# + Method E: Popularity-aware negative sampling (hard negatives).
# + Added evaluation metrics:
#   1) ItemCoverage@K
#   2) Novelty: AvgLogPopularity@K and AvgSelfInformation@K
#   3) Credibility-weighted utility (CredUtility@K) + High/Low credibility group Recall@K (top 20% vs bottom 20%)

import csv
import json
import math
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    jsonl_path: str = str((SCRIPT_DIR / "dataset" / "Clothing_Shoes_and_Jewelry.jsonl").resolve())
    out_dir: str = str((SCRIPT_DIR / "dataset" / "lightgcn_pipeline_cu_message_parent_pop").resolve())

    user_key: str = "user_id"
    item_key: str = "parent_asin"
    rating_key: str = "rating"

    pos_rating_threshold: float = 4.0

    train_p: float = 0.80
    val_p: float = 0.10
    test_p: float = 0.10

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dim: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    reg: float = 1e-4
    epochs: int = 400
    batch_size: int = 4096

    Ks: tuple = (10, 20)
    eval_every: int = 1

    eval_mode: str = "sampled"   # "sampled" or "full"
    sampled_negatives: int = 99

    print_every: int = 1_000_000
    decode_errors: str = "replace"

    # ✅ credibility CSV path (minmax)
    cred_csv_path: str = str(
        (SCRIPT_DIR / "dataset" / "graph_pyg_parent_asin" / "credibility_scores_minmax.csv").resolve()
    )

    # =========================
    # Method E: popularity-aware negative sampling
    # =========================
    neg_mix_pop: float = 0.7      # mix weight for popularity-based sampling (0..1). Try 0.6–0.9
    neg_pop_gamma: float = 0.75   # exponent on degree (common: 0.75). Try 0.5–1.0
    neg_max_tries: int = 50       # avoid infinite loops

    # =========================
    # Cred group split
    # =========================
    cred_group_pct: float = 0.20  # top/bottom 20%


cfg = CFG()


# =========================
# UTIL
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def is_positive_interaction(rec: dict) -> bool:
    uid = rec.get(cfg.user_key)
    iid = rec.get(cfg.item_key)
    if uid is None or iid is None:
        return False
    rating = to_float(rec.get(cfg.rating_key))
    if rating is None:
        return False
    return rating >= cfg.pos_rating_threshold


def split_bucket(uid: str, iid: str) -> str:
    s = f"{uid}|{iid}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < cfg.train_p:
        return "train"
    elif x < cfg.train_p + cfg.val_p:
        return "val"
    else:
        return "test"


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_paths():
    jsonl = Path(cfg.jsonl_path)
    if not jsonl.exists():
        raise FileNotFoundError(
            f"JSONL not found:\n  {jsonl}\n\n"
            f"Current working dir: {Path.cwd()}\n"
            f"Put the file under: {SCRIPT_DIR / 'dataset'}\n"
            f"Or set cfg.jsonl_path to your absolute JSONL path.\n"
        )


def iter_jsonl_records(path: Path):
    bad_json = 0
    total = 0

    with open(path, "rb") as f:
        for raw in f:
            total += 1
            line = raw.decode("utf-8", errors=cfg.decode_errors).strip()
            if not line:
                continue
            try:
                yield total, json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                if bad_json <= 5:
                    print(f"[WARN] Skipping invalid JSON at line {total}")
                continue

    if bad_json > 0:
        print(f"[WARN] Total invalid JSON lines skipped: {bad_json:,}")


# =========================
# CREDIBILITY LOADER
# =========================
def load_credibility_vector(cred_csv_path: str, user2idx: dict) -> np.ndarray:
    num_users = len(user2idx)
    cred = np.ones((num_users,), dtype=np.float32)

    p = Path(cred_csv_path)
    if not p.exists():
        print(f"[CRED] Cred CSV not found: {p} -> using all ones.")
        return cred

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        cols = set(headers)

        used, skipped = 0, 0

        if "user_id" in cols and "credibility" in cols:
            for row in reader:
                uid = row.get("user_id")
                if not uid:
                    continue
                uidx = user2idx.get(uid)
                if uidx is None:
                    skipped += 1
                    continue
                try:
                    cred[uidx] = float(row["credibility"])
                    used += 1
                except Exception:
                    continue
            print(f"[CRED] Loaded by user_id. used={used:,} skipped_not_in_lightgcn={skipped:,}")

        elif "user_idx" in cols and "credibility" in cols:
            for row in reader:
                try:
                    u = int(row["user_idx"])
                    if 0 <= u < num_users:
                        cred[u] = float(row["credibility"])
                        used += 1
                except Exception:
                    continue
            print(f"[CRED] Loaded by user_idx. used={used:,}")

        else:
            raise ValueError(
                f"[CRED] Unsupported cred CSV header: {headers}. "
                f"Expected (user_id,credibility) OR (user_idx,credibility)."
            )

    cred = np.clip(cred, 0.0, 1.0).astype(np.float32)
    p10, p50, p90, p99 = np.percentile(cred, [10, 50, 90, 99])
    print(
        f"[CRED] stats: min={cred.min():.4f} p10={p10:.4f} p50={p50:.4f} "
        f"p90={p90:.4f} p99={p99:.4f} max={cred.max():.4f}"
    )
    return cred


# =========================
# STEP 1: GRAPH CONSTRUCTION (unchanged)
# =========================
def build_graph_from_jsonl():
    ensure_paths()

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "npy").mkdir(parents=True, exist_ok=True)

    user2idx = {}
    item2idx = {}
    counts = {"train": 0, "val": 0, "test": 0}
    pos_edges = 0

    jsonl_path = Path(cfg.jsonl_path)

    for i, rec in iter_jsonl_records(jsonl_path):
        if not is_positive_interaction(rec):
            continue

        uid = rec[cfg.user_key]
        iid = rec[cfg.item_key]

        if uid not in user2idx:
            user2idx[uid] = len(user2idx)
        if iid not in item2idx:
            item2idx[iid] = len(item2idx)

        b = split_bucket(uid, iid)
        counts[b] += 1
        pos_edges += 1

        if i % cfg.print_every == 0:
            print(
                f"PASS1 {i:,} lines | users={len(user2idx):,} items={len(item2idx):,} "
                f"pos_edges={pos_edges:,} train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}"
            )

    print("\nPASS1 done.")
    print("Users:", len(user2idx), "Items:", len(item2idx), "Positive edges:", pos_edges)
    print("Split counts:", counts)

    save_pickle(user2idx, out / "model" / "user2idx.pkl")
    save_pickle(item2idx, out / "model" / "item2idx.pkl")

    train = np.empty((2, counts["train"]), dtype=np.int32)
    val = np.empty((2, counts["val"]), dtype=np.int32)
    test = np.empty((2, counts["test"]), dtype=np.int32)
    ptr = {"train": 0, "val": 0, "test": 0}

    for _, rec in iter_jsonl_records(jsonl_path):
        if not is_positive_interaction(rec):
            continue

        uid = rec[cfg.user_key]
        iid = rec[cfg.item_key]
        u = user2idx[uid]
        it = item2idx[iid]

        b = split_bucket(uid, iid)
        p = ptr[b]
        if b == "train":
            train[0, p] = u
            train[1, p] = it
        elif b == "val":
            val[0, p] = u
            val[1, p] = it
        else:
            test[0, p] = u
            test[1, p] = it

        ptr[b] += 1

    np.save(out / "npy" / "train_edges.npy", train)
    np.save(out / "npy" / "val_edges.npy", val)
    np.save(out / "npy" / "test_edges.npy", test)

    print("\n✅ Saved graph files to:", out)


# =========================
# CSR helpers (unchanged)
# =========================
def edges_to_user_csr(edges_2xE: np.ndarray, num_users: int):
    u = edges_2xE[0].astype(np.int64)
    it = edges_2xE[1].astype(np.int64)

    order = np.argsort(u, kind="mergesort")
    u = u[order]
    it = it[order]

    counts = np.bincount(u, minlength=num_users)
    indptr = np.zeros(num_users + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    indices = it.copy()
    for user in range(num_users):
        start, end = indptr[user], indptr[user + 1]
        if end - start > 1:
            indices[start:end] = np.sort(indices[start:end])

    return indptr, indices


def user_has_item(indptr, indices, user: int, item: int) -> bool:
    start, end = indptr[user], indptr[user + 1]
    if start == end:
        return False
    arr = indices[start:end]
    j = np.searchsorted(arr, item)
    return j < (end - start) and arr[j] == item


def sample_pos_item(indptr, indices, user: int, rng: np.random.Generator):
    start, end = indptr[user], indptr[user + 1]
    if start == end:
        return None
    return int(indices[rng.integers(start, end)])


# =========================
# Method E: popularity-aware negative sampling
# =========================
def sample_neg_item_popmix(
    indptr,
    indices,
    user: int,
    num_items: int,
    rng: np.random.Generator,
    pop_prob: np.ndarray,
    mix_pop: float,
    max_tries: int,
):
    """
    With probability mix_pop: sample from pop_prob (hard negatives, popularity-based)
    Otherwise: uniform.
    Reject if user already interacted with it.
    """
    for _ in range(max_tries):
        if rng.random() < mix_pop:
            j = int(rng.choice(num_items, p=pop_prob))
        else:
            j = int(rng.integers(0, num_items))

        if not user_has_item(indptr, indices, user, j):
            return j

    while True:
        j = int(rng.integers(0, num_items))
        if not user_has_item(indptr, indices, user, j):
            return j


# =========================
# Popularity + Novelty helpers (NEW)
# =========================
def compute_item_popularity(train_edges_2xE: np.ndarray, num_items: int):
    """pop[i] = count of item i in TRAIN edges."""
    items = train_edges_2xE[1].astype(np.int64)
    pop = np.bincount(items, minlength=num_items).astype(np.int64)
    total = int(pop.sum())
    return pop, total


def novelty_stats_for_items(item_ids, pop: np.ndarray, total_train: int, num_items: int):
    """
    avg_log_popularity = mean(log(pop+1))   (lower => more novel)
    avg_self_information = mean(-log2(p(i))) with Laplace smoothing (higher => more novel)
    """
    item_ids = np.asarray(item_ids, dtype=np.int64)
    if item_ids.size == 0:
        return 0.0, 0.0

    pops = pop[item_ids]
    avg_log_popularity = float(np.log(pops + 1.0).mean())

    p = (pops + 1.0) / (total_train + num_items)  # Laplace smoothing
    avg_self_information = float((-np.log2(p)).mean())
    return avg_log_popularity, avg_self_information


def make_cred_groups(users: np.ndarray, cred: np.ndarray, pct: float):
    """
    users: subset of users to evaluate
    cred:  full cred vector [num_users]
    pct:   top/bottom fraction
    """
    if users.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    c = cred[users]
    n = users.size
    k = max(int(round(n * pct)), 1)

    order = np.argsort(c)  # ascending
    low = users[order[:k]]
    high = users[order[-k:]]
    return high.astype(np.int64), low.astype(np.int64)


# =========================
# Build credibility-weighted propagation matrices (unchanged)
# =========================
def build_message_passing_mats(train_edges_2xE: np.ndarray, num_users: int, num_items: int, cred_u: torch.Tensor, device: str):
    u = train_edges_2xE[0].astype(np.int64)
    i = train_edges_2xE[1].astype(np.int64)

    deg_u = np.bincount(u, minlength=num_users).astype(np.float32)
    deg_i = np.bincount(i, minlength=num_items).astype(np.float32)

    inv_sqrt_u = 1.0 / np.sqrt(np.maximum(deg_u, 1.0))
    inv_sqrt_i = 1.0 / np.sqrt(np.maximum(deg_i, 1.0))

    w_base = inv_sqrt_u[u] * inv_sqrt_i[i]

    idx_ui = torch.tensor(np.vstack([u, i]), dtype=torch.long, device=device)
    val_ui = torch.tensor(w_base, dtype=torch.float32, device=device)
    M_ui = torch.sparse_coo_tensor(idx_ui, val_ui, size=(num_users, num_items)).coalesce()

    c_u_np = cred_u.detach().view(-1).cpu().numpy().astype(np.float32)
    w_cred = c_u_np[u] * w_base

    idx_iu = torch.tensor(np.vstack([i, u]), dtype=torch.long, device=device)
    val_iu = torch.tensor(w_cred, dtype=torch.float32, device=device)
    M_iu = torch.sparse_coo_tensor(idx_iu, val_iu, size=(num_items, num_users)).coalesce()

    return M_ui, M_iu


# =========================
# LightGCN (unchanged)
# =========================
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_layers, M_ui: torch.Tensor, M_iu: torch.Tensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.M_ui = M_ui
        self.M_iu = M_iu

        self.user_emb = torch.nn.Embedding(num_users, emb_dim)
        self.item_emb = torch.nn.Embedding(num_items, emb_dim)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self):
        u0 = self.user_emb.weight
        i0 = self.item_emb.weight

        u_list = [u0]
        i_list = [i0]

        u = u0
        i = i0

        for _ in range(self.num_layers):
            i = torch.sparse.mm(self.M_iu, u)
            u = torch.sparse.mm(self.M_ui, i)
            u_list.append(u)
            i_list.append(i)

        u_final = torch.stack(u_list, dim=0).mean(dim=0)
        i_final = torch.stack(i_list, dim=0).mean(dim=0)
        return u_final, i_final

    def get_user_item_emb(self):
        return self.propagate()

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb, reg_weight: float):
        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

        reg = (
            self.user_emb.weight[users].norm(2, dim=1).pow(2)
            + self.item_emb.weight[pos_items].norm(2, dim=1).pow(2)
            + self.item_emb.weight[neg_items].norm(2, dim=1).pow(2)
        ).mean()
        return loss + reg_weight * reg


# =========================
# Base ranking metrics (unchanged)
# =========================
def metrics_at_k(ranked_items, gt_set, K):
    topk = ranked_items[:K]
    hits = [1 if x in gt_set else 0 for x in topk]
    hit_count = sum(hits)

    precision = hit_count / K
    recall = hit_count / max(len(gt_set), 1)

    dcg = 0.0
    for idx, h in enumerate(hits):
        if h:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(gt_set), K)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, ndcg


# =========================
# Evaluation with Coverage/Novelty/CredUtility + Cred-groups (NEW)
# =========================
@torch.no_grad()
def evaluate_sampled(
    model: LightGCN,
    train_csr,
    test_csr,
    num_items: int,
    device: str,
    item_pop: np.ndarray,
    total_train_interactions: int,
    cred_np: np.ndarray,
):
    indptr_tr, indices_tr = train_csr
    indptr_te, indices_te = test_csr

    user_emb, item_emb = model.get_user_item_emb()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    rng = np.random.default_rng(cfg.seed + 999)

    users = np.where((indptr_te[1:] - indptr_te[:-1]) > 0)[0]
    if len(users) == 0:
        raise RuntimeError("No users with test interactions. Check your split or threshold.")

    users = users.astype(np.int64)

    # groups among evaluated users
    high_users, low_users = make_cred_groups(users, cred_np, cfg.cred_group_pct)
    high_set = set(map(int, high_users.tolist()))
    low_set = set(map(int, low_users.tolist()))

    sums = {K: {"p": 0.0, "r": 0.0, "n": 0.0, "logpop": 0.0, "selfinfo": 0.0} for K in cfg.Ks}
    rec_items = {K: set() for K in cfg.Ks}

    cred_sum = 0.0
    n_users = 0

    grp = {K: {"high_r": 0.0, "low_r": 0.0, "high_n": 0, "low_n": 0} for K in cfg.Ks}

    for u in users:
        start, end = indptr_te[u], indptr_te[u + 1]
        gt_items = indices_te[start:end]
        gt_set = set(map(int, gt_items.tolist()))

        pos = int(gt_items[rng.integers(0, len(gt_items))])

        negs = []
        while len(negs) < cfg.sampled_negatives:
            j = int(rng.integers(0, num_items))
            if j in gt_set:
                continue
            if user_has_item(indptr_tr, indices_tr, int(u), j):
                continue
            negs.append(j)

        cand = np.array([pos] + negs, dtype=np.int64)

        uvec = user_emb[int(u)].unsqueeze(0)
        ivec = item_emb[cand]
        scores = (uvec * ivec).sum(dim=1).detach().cpu().numpy()

        ranked = cand[np.argsort(-scores)]

        cred_sum += float(cred_np[int(u)])

        for K in cfg.Ks:
            topk = ranked[:K]

            p, r, n = metrics_at_k(ranked, {pos}, K)
            sums[K]["p"] += p
            sums[K]["r"] += r
            sums[K]["n"] += n

            # coverage
            rec_items[K].update(map(int, topk.tolist()))

            # novelty
            lp, si = novelty_stats_for_items(topk, item_pop, total_train_interactions, num_items)
            sums[K]["logpop"] += lp
            sums[K]["selfinfo"] += si

            # group recall (sampled recall is 0/1)
            if int(u) in high_set:
                grp[K]["high_r"] += r
                grp[K]["high_n"] += 1
            if int(u) in low_set:
                grp[K]["low_r"] += r
                grp[K]["low_n"] += 1

        n_users += 1

    results = {}
    for K in cfg.Ks:
        results[K] = {
            "precision": sums[K]["p"] / n_users,
            "recall": sums[K]["r"] / n_users,
            "ndcg": sums[K]["n"] / n_users,

            "item_coverage": len(rec_items[K]) / max(num_items, 1),
            "avg_log_popularity": sums[K]["logpop"] / n_users,
            "avg_self_information": sums[K]["selfinfo"] / n_users,

            # your CredUtility formula simplifies to mean(c_u) across evaluated users
            "cred_utility": cred_sum / n_users,

            "high_cred_recall": grp[K]["high_r"] / max(grp[K]["high_n"], 1),
            "low_cred_recall": grp[K]["low_r"] / max(grp[K]["low_n"], 1),
            "high_users": int(grp[K]["high_n"]),
            "low_users": int(grp[K]["low_n"]),

            "users_eval": n_users,
            "mode": "sampled(1pos+neg)",
            "negatives": cfg.sampled_negatives,
        }
    return results


@torch.no_grad()
def evaluate_full_ranking(
    model: LightGCN,
    train_csr,
    test_csr,
    num_items: int,
    device: str,
    item_pop: np.ndarray,
    total_train_interactions: int,
    cred_np: np.ndarray,
):
    indptr_tr, indices_tr = train_csr
    indptr_te, indices_te = test_csr

    user_emb, item_emb = model.get_user_item_emb()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    users = np.where((indptr_te[1:] - indptr_te[:-1]) > 0)[0]
    if len(users) == 0:
        raise RuntimeError("No users with test interactions. Check your split or threshold.")

    users = users.astype(np.int64)

    high_users, low_users = make_cred_groups(users, cred_np, cfg.cred_group_pct)
    high_set = set(map(int, high_users.tolist()))
    low_set = set(map(int, low_users.tolist()))

    sums = {K: {"p": 0.0, "r": 0.0, "n": 0.0, "logpop": 0.0, "selfinfo": 0.0} for K in cfg.Ks}
    rec_items = {K: set() for K in cfg.Ks}

    cred_sum = 0.0
    n_users = 0

    grp = {K: {"high_r": 0.0, "low_r": 0.0, "high_n": 0, "low_n": 0} for K in cfg.Ks}

    all_items = torch.arange(num_items, device=device)

    for u in users:
        start, end = indptr_te[u], indptr_te[u + 1]
        gt = indices_te[start:end]
        gt_set = set(map(int, gt.tolist()))

        uvec = user_emb[int(u)].unsqueeze(0)
        scores = (uvec * item_emb).sum(dim=1)

        tr_s, tr_e = indptr_tr[u], indptr_tr[u + 1]
        train_items = indices_tr[tr_s:tr_e]
        if len(train_items) > 0:
            scores[torch.tensor(train_items, device=device, dtype=torch.long)] = -1e9

        ranked = all_items[torch.argsort(scores, descending=True)].detach().cpu().numpy()

        cred_sum += float(cred_np[int(u)])

        for K in cfg.Ks:
            topk = ranked[:K]

            p, r, n = metrics_at_k(ranked, gt_set, K)
            sums[K]["p"] += p
            sums[K]["r"] += r
            sums[K]["n"] += n

            rec_items[K].update(map(int, topk.tolist()))

            lp, si = novelty_stats_for_items(topk, item_pop, total_train_interactions, num_items)
            sums[K]["logpop"] += lp
            sums[K]["selfinfo"] += si

            if int(u) in high_set:
                grp[K]["high_r"] += r
                grp[K]["high_n"] += 1
            if int(u) in low_set:
                grp[K]["low_r"] += r
                grp[K]["low_n"] += 1

        n_users += 1

    results = {}
    for K in cfg.Ks:
        results[K] = {
            "precision": sums[K]["p"] / n_users,
            "recall": sums[K]["r"] / n_users,
            "ndcg": sums[K]["n"] / n_users,

            "item_coverage": len(rec_items[K]) / max(num_items, 1),
            "avg_log_popularity": sums[K]["logpop"] / n_users,
            "avg_self_information": sums[K]["selfinfo"] / n_users,

            "cred_utility": cred_sum / n_users,

            "high_cred_recall": grp[K]["high_r"] / max(grp[K]["high_n"], 1),
            "low_cred_recall": grp[K]["low_r"] / max(grp[K]["low_n"], 1),
            "high_users": int(grp[K]["high_n"]),
            "low_users": int(grp[K]["low_n"]),

            "users_eval": n_users,
            "mode": "full",
        }
    return results


# =========================
# TRAIN LOOP
# =========================
def train_lightgcn():
    out = Path(cfg.out_dir)

    train_edges = np.load(out / "npy" / "train_edges.npy")
    val_edges = np.load(out / "npy" / "val_edges.npy")
    test_edges = np.load(out / "npy" / "test_edges.npy")
    user2idx = load_pickle(out / "model" / "user2idx.pkl")
    item2idx = load_pickle(out / "model" / "item2idx.pkl")

    num_users = len(user2idx)
    num_items = len(item2idx)

    print(
        f"Loaded edges. Users={num_users:,} Items={num_items:,} "
        f"Train={train_edges.shape[1]:,} Val={val_edges.shape[1]:,} Test={test_edges.shape[1]:,}"
    )

    train_csr = edges_to_user_csr(train_edges, num_users)
    val_csr = edges_to_user_csr(val_edges, num_users)
    test_csr = edges_to_user_csr(test_edges, num_users)

    device = cfg.device
    print("Using device:", device)

    # NEW: popularity (from TRAIN)
    item_pop, total_train_interactions = compute_item_popularity(train_edges, num_items)

    # Load credibility
    cred_np = load_credibility_vector(cfg.cred_csv_path, user2idx)
    cred_t = torch.tensor(cred_np, dtype=torch.float32, device=device)

    # Build message-passing mats (credibility inside user->item)
    M_ui, M_iu = build_message_passing_mats(train_edges, num_users, num_items, cred_t, device=device)

    model = LightGCN(num_users, num_items, cfg.emb_dim, cfg.num_layers, M_ui, M_iu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rng = np.random.default_rng(cfg.seed)

    indptr_tr, _ = train_csr
    train_users = np.where((indptr_tr[1:] - indptr_tr[:-1]) > 0)[0]
    if len(train_users) == 0:
        raise RuntimeError("No train users with interactions. Check your threshold/split.")

    # =========================
    # Method E: build popularity distribution from TRAIN degrees (already here)
    # =========================
    train_item_ids = train_edges[1].astype(np.int64)
    item_deg = np.bincount(train_item_ids, minlength=num_items).astype(np.float64)

    pop = np.power(item_deg + 1.0, cfg.neg_pop_gamma)
    pop_prob = pop / (pop.sum() + 1e-12)
    pop_prob = pop_prob.astype(np.float64)

    p10, p50, p90, p99 = np.percentile(item_deg, [10, 50, 90, 99])
    print(f"[NEG-E] item_deg percentiles: p10={p10:.0f} p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} max={item_deg.max():.0f}")
    print(f"[NEG-E] mix_pop={cfg.neg_mix_pop} gamma={cfg.neg_pop_gamma}")

    best_val = -1.0
    best_path = out / "model" / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(train_users)

        total_loss = 0.0
        steps = 0

        for start in range(0, len(train_users), cfg.batch_size):
            batch_users = train_users[start : start + cfg.batch_size]

            pos_items = []
            neg_items = []
            used_users = []

            indptr, indices = train_csr

            for u in batch_users:
                p = sample_pos_item(indptr, indices, int(u), rng)
                if p is None:
                    continue

                n = sample_neg_item_popmix(
                    indptr, indices, int(u), num_items, rng,
                    pop_prob=pop_prob,
                    mix_pop=cfg.neg_mix_pop,
                    max_tries=cfg.neg_max_tries
                )

                used_users.append(int(u))
                pos_items.append(p)
                neg_items.append(n)

            if len(pos_items) == 0:
                continue

            users_t = torch.tensor(used_users, device=device, dtype=torch.long)
            pos_t = torch.tensor(pos_items, device=device, dtype=torch.long)
            neg_t = torch.tensor(neg_items, device=device, dtype=torch.long)

            user_emb, item_emb = model.get_user_item_emb()
            loss = model.bpr_loss(users_t, pos_t, neg_t, user_emb, item_emb, cfg.reg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.6f}")

        if epoch % cfg.eval_every == 0:
            model.eval()

            if cfg.eval_mode == "full":
                val_res = evaluate_full_ranking(
                    model, train_csr, val_csr, num_items, device,
                    item_pop, total_train_interactions, cred_np
                )
            else:
                val_res = evaluate_sampled(
                    model, train_csr, val_csr, num_items, device,
                    item_pop, total_train_interactions, cred_np
                )

            selK = max(cfg.Ks)
            val_score = val_res[selK]["recall"]

            print("VAL metrics:")
            for K in cfg.Ks:
                r = val_res[K]
                print(
                    f"  K={K}: "
                    f"P={r['precision']:.4f} R={r['recall']:.4f} NDCG={r['ndcg']:.4f} "
                    f"COV={r['item_coverage']:.4f} "
                    f"LogPop={r['avg_log_popularity']:.4f} SI={r['avg_self_information']:.4f} "
                    f"CredU={r['cred_utility']:.4f} "
                    f"HighR={r['high_cred_recall']:.4f} LowR={r['low_cred_recall']:.4f} "
                    f"({r['mode']})"
                )

            if val_score > best_val:
                best_val = val_score
                torch.save(model.state_dict(), best_path)
                print(f"  ✅ Saved best model to {best_path} (val Recall@{selK}={best_val:.4f})")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

    if cfg.eval_mode == "full":
        test_res = evaluate_full_ranking(
            model, train_csr, test_csr, num_items, device,
            item_pop, total_train_interactions, cred_np
        )
    else:
        test_res = evaluate_sampled(
            model, train_csr, test_csr, num_items, device,
            item_pop, total_train_interactions, cred_np
        )

    print("\nTEST metrics:")
    for K in cfg.Ks:
        r = test_res[K]
        print(
            f"  K={K}: "
            f"P={r['precision']:.4f} R={r['recall']:.4f} NDCG={r['ndcg']:.4f} "
            f"COV={r['item_coverage']:.4f} "
            f"LogPop={r['avg_log_popularity']:.4f} SI={r['avg_self_information']:.4f} "
            f"CredU={r['cred_utility']:.4f} "
            f"HighR={r['high_cred_recall']:.4f} LowR={r['low_cred_recall']:.4f} "
            f"({r['mode']})"
        )

    return test_res


def main():
    set_seed(cfg.seed)
    ensure_paths()

    out = Path(cfg.out_dir)
    train_edge_path = out / "npy" / "train_edges.npy"

    if not train_edge_path.exists():
        print("Graph files not found. Building graph first...")
        build_graph_from_jsonl()
    else:
        print("Graph files exist. Skipping construction.")

    train_lightgcn()


if __name__ == "__main__":
    main()
