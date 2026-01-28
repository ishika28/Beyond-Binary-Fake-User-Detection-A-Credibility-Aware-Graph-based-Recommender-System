# lightgcn_full_pipeline.py
# Build graph (user-item edges) from JSONL -> split -> train LightGCN -> evaluate Recall/Precision/NDCG.

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
    out_dir: str = str((SCRIPT_DIR / "dataset" / "lightgcn_pipeline_parent").resolve())

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
    epochs: int = 200
    batch_size: int = 4096

    Ks: tuple = (10, 20)
    eval_every: int = 1

    eval_mode: str = "sampled"
    sampled_negatives: int = 99

    print_every: int = 1_000_000

    # decoding safety
    decode_errors: str = "replace"   # "replace" (recommended) or "ignore"


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
    """
    Streaming JSONL reader that tolerates non-UTF8 bytes.
    - Reads bytes -> decodes with errors=replace/ignore
    - Skips invalid JSON lines
    """
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
                # skip line; continue streaming
                if bad_json <= 5:
                    print(f"[WARN] Skipping invalid JSON at line {total}")
                continue

    if bad_json > 0:
        print(f"[WARN] Total invalid JSON lines skipped: {bad_json:,}")


# =========================
# STEP 1: GRAPH CONSTRUCTION
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

    # PASS 1
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

    # PASS 2
    train = np.empty((2, counts["train"]), dtype=np.int32)
    val = np.empty((2, counts["val"]), dtype=np.int32)
    test = np.empty((2, counts["test"]), dtype=np.int32)
    ptr = {"train": 0, "val": 0, "test": 0}

    for i, rec in iter_jsonl_records(jsonl_path):
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

        if i % cfg.print_every == 0:
            print(
                f"PASS2 {i:,} lines | wrote train={ptr['train']:,}/{counts['train']:,} "
                f"val={ptr['val']:,}/{counts['val']:,} test={ptr['test']:,}/{counts['test']:,}"
            )

    assert ptr["train"] == counts["train"]
    assert ptr["val"] == counts["val"]
    assert ptr["test"] == counts["test"]

    np.save(out / "npy" / "train_edges.npy", train)
    np.save(out / "npy" / "val_edges.npy", val)
    np.save(out / "npy" / "test_edges.npy", test)

    meta = {
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "pos_rating_threshold": cfg.pos_rating_threshold,
        "split": {"train": cfg.train_p, "val": cfg.val_p, "test": cfg.test_p},
        "counts": counts,
        "item_key": cfg.item_key,
        "user_key": cfg.user_key,
        "jsonl_path": str(jsonl_path),
        "decode_errors": cfg.decode_errors,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Saved graph files to:", out)


# =========================
# CSR helpers
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


def sample_neg_item(indptr, indices, user: int, num_items: int, rng: np.random.Generator):
    while True:
        j = int(rng.integers(0, num_items))
        if not user_has_item(indptr, indices, user, j):
            return j


# =========================
# LightGCN
# =========================
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_layers, norm_adj: torch.Tensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers
        self.norm_adj = norm_adj

        self.emb = torch.nn.Embedding(self.num_nodes, emb_dim)
        torch.nn.init.xavier_uniform_(self.emb.weight)

    def propagate(self):
        x0 = self.emb.weight
        xs = [x0]
        x = x0
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            xs.append(x)
        return torch.stack(xs, dim=0).mean(dim=0)

    def get_user_item_emb(self):
        x_final = self.propagate()
        user_emb = x_final[: self.num_users]
        item_emb = x_final[self.num_users :]
        return user_emb, item_emb

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb, reg_weight: float):
        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

        ego_u = self.emb.weight[users]
        ego_p = self.emb.weight[self.num_users + pos_items]
        ego_n = self.emb.weight[self.num_users + neg_items]
        reg = (
            ego_u.norm(2, dim=1).pow(2)
            + ego_p.norm(2, dim=1).pow(2)
            + ego_n.norm(2, dim=1).pow(2)
        ).mean()
        return loss + reg_weight * reg


def build_norm_adj(train_edges, num_users, num_items, device):
    u = train_edges[0].astype(np.int64)
    it = train_edges[1].astype(np.int64) + num_users

    row = np.concatenate([u, it])
    col = np.concatenate([it, u])
    data = np.ones_like(row, dtype=np.float32)

    N = num_users + num_items
    idx = torch.tensor(np.vstack([row, col]), dtype=torch.long, device=device)
    val = torch.tensor(data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(idx, val, size=(N, N)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    r, c = adj.indices()
    v = adj.values()
    v = v * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    return torch.sparse_coo_tensor(adj.indices(), v, size=adj.size()).coalesce()


# =========================
# Evaluation
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


@torch.no_grad()
def evaluate_sampled(model: LightGCN, train_csr, test_csr, num_items: int, device: str):
    indptr_tr, indices_tr = train_csr
    indptr_te, indices_te = test_csr

    user_emb, item_emb = model.get_user_item_emb()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    rng = np.random.default_rng(cfg.seed + 999)

    users = np.where((indptr_te[1:] - indptr_te[:-1]) > 0)[0]
    if len(users) == 0:
        raise RuntimeError("No users with test interactions. Check your split or threshold.")

    sums = {K: {"p": 0.0, "r": 0.0, "n": 0.0} for K in cfg.Ks}
    n_users = 0

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
        for K in cfg.Ks:
            p, r, n = metrics_at_k(ranked, {pos}, K)
            sums[K]["p"] += p
            sums[K]["r"] += r
            sums[K]["n"] += n

        n_users += 1

    results = {}
    for K in cfg.Ks:
        results[K] = {
            "precision": sums[K]["p"] / n_users,
            "recall": sums[K]["r"] / n_users,
            "ndcg": sums[K]["n"] / n_users,
            "users_eval": n_users,
            "mode": "sampled(1pos+neg)",
            "negatives": cfg.sampled_negatives,
        }
    return results


@torch.no_grad()
def evaluate_full_ranking(model: LightGCN, train_csr, test_csr, num_items: int, device: str):
    indptr_tr, indices_tr = train_csr
    indptr_te, indices_te = test_csr

    user_emb, item_emb = model.get_user_item_emb()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    users = np.where((indptr_te[1:] - indptr_te[:-1]) > 0)[0]
    if len(users) == 0:
        raise RuntimeError("No users with test interactions. Check your split or threshold.")

    sums = {K: {"p": 0.0, "r": 0.0, "n": 0.0} for K in cfg.Ks}
    n_users = 0

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

        for K in cfg.Ks:
            p, r, n = metrics_at_k(ranked, gt_set, K)
            sums[K]["p"] += p
            sums[K]["r"] += r
            sums[K]["n"] += n

        n_users += 1

    results = {}
    for K in cfg.Ks:
        results[K] = {
            "precision": sums[K]["p"] / n_users,
            "recall": sums[K]["r"] / n_users,
            "ndcg": sums[K]["n"] / n_users,
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

    norm_adj = build_norm_adj(train_edges, num_users, num_items, device=device)

    model = LightGCN(num_users, num_items, cfg.emb_dim, cfg.num_layers, norm_adj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rng = np.random.default_rng(cfg.seed)

    indptr_tr, _ = train_csr
    train_users = np.where((indptr_tr[1:] - indptr_tr[:-1]) > 0)[0]
    if len(train_users) == 0:
        raise RuntimeError("No train users with interactions. Check your threshold/split.")

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

            indptr, indices = train_csr

            for u in batch_users:
                p = sample_pos_item(indptr, indices, int(u), rng)
                if p is None:
                    continue
                pos_items.append(p)
                n = sample_neg_item(indptr, indices, int(u), num_items, rng)
                neg_items.append(n)

            if len(pos_items) == 0:
                continue

            users_t = torch.tensor(batch_users[: len(pos_items)], device=device, dtype=torch.long)
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
                val_res = evaluate_full_ranking(model, train_csr, val_csr, num_items, device)
            else:
                val_res = evaluate_sampled(model, train_csr, val_csr, num_items, device)

            selK = max(cfg.Ks)
            val_score = val_res[selK]["recall"]

            print("VAL metrics:")
            for K in cfg.Ks:
                r = val_res[K]
                print(f"  K={K}: P={r['precision']:.4f} R={r['recall']:.4f} NDCG={r['ndcg']:.4f} ({r['mode']})")

            if val_score > best_val:
                best_val = val_score
                torch.save(model.state_dict(), best_path)
                print(f"  ✅ Saved best model to {best_path} (val Recall@{selK}={best_val:.4f})")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

    if cfg.eval_mode == "full":
        test_res = evaluate_full_ranking(model, train_csr, test_csr, num_items, device)
    else:
        test_res = evaluate_sampled(model, train_csr, test_csr, num_items, device)

    print("\nTEST metrics:")
    for K in cfg.Ks:
        r = test_res[K]
        print(f"  K={K}: P={r['precision']:.4f} R={r['recall']:.4f} NDCG={r['ndcg']:.4f} ({r['mode']})")

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
