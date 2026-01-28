# lightgcn_full_pipeline_cred_all_eq_3_22_to_3_28.py
# Raw JSONL -> split -> train credibility-weighted LightGCN -> eval.
#
# Implements Thesis Eq (3.22)–(3.28):
# (3.22) init embeddings
# (3.23) user->item message passing weighted by credibility c_hat[u]
# (3.24) item->user message passing standard
# (3.25) average embeddings across layers
# (3.26) prediction y_hat = e_u^T e_i
# (3.27) L_fair = sum pop(i) * y_hat(u,i)
# (3.28) L = lambda_fair * L_fair + lambda_reg * L_reg   (we add BPR for ranking learning)

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
    out_dir: str = str((SCRIPT_DIR / "dataset" / "lightgcn_pipeline_parent").resolve())

    # credibility CSV
    # supports:
    #   - user_id,credibility
    #   - user_idx,credibility
    cred_csv_path: str = str((SCRIPT_DIR / "dataset" / "graph_pyg_parent_asin" / "credibility_scores_minmax.csv").resolve())

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

    # Eq (3.28): lambda_reg * Lreg
    lambda_reg: float = 1e-4

    # Eq (3.28): lambda_fair * Lfair
    lambda_fair: float = 0.0  # set e.g. 1e-2 to enable Eq (3.27)

    epochs: int = 400
    batch_size: int = 4096

    Ks: tuple = (10, 20)
    eval_every: int = 1

    eval_mode: str = "sampled"      # "sampled" or "full"
    sampled_negatives: int = 99

    print_every: int = 1_000_000

    decode_errors: str = "replace"


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
                f"PASS1 {i:,} | users={len(user2idx):,} items={len(item2idx):,} "
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
                f"PASS2 {i:,} | train={ptr['train']:,}/{counts['train']:,} "
                f"val={ptr['val']:,}/{counts['val']:,} test={ptr['test']:,}/{counts['test']:,}"
            )

    assert ptr["train"] == counts["train"]
    assert ptr["val"] == counts["val"]
    assert ptr["test"] == counts["test"]

    np.save(out / "npy" / "train_edges.npy", train)
    np.save(out / "npy" / "val_edges.npy", val)
    np.save(out / "npy" / "test_edges.npy", test)

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
# Credibility loader
# =========================
def load_credibility_vector(num_users: int, user2idx: dict) -> np.ndarray:
    """
    Returns cred[num_users] float32 (clamped 0..1)
    Missing users -> default 1.0
    CSV formats:
      (A) user_id,credibility
      (B) user_idx,credibility
    """
    cred = np.ones((num_users,), dtype=np.float32)
    p = Path(cfg.cred_csv_path)

    if not p.exists():
        print(f"[CRED] Cred CSV not found: {p}. Using all-ones credibility.")
        return cred

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        cols_set = set(cols)

        used, skipped = 0, 0

        if "user_id" in cols_set and "credibility" in cols_set:
            for row in reader:
                uid = row.get("user_id")
                if not uid:
                    continue
                if uid not in user2idx:
                    skipped += 1
                    continue
                try:
                    cred[user2idx[uid]] = float(row["credibility"])
                    used += 1
                except Exception:
                    continue
            print(f"[CRED] Loaded by user_id. used={used:,} skipped_not_in_lightgcn={skipped:,}")

        elif "user_idx" in cols_set and "credibility" in cols_set:
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
                f"[CRED] Unsupported cred CSV header: {cols}. "
                f"Expected (user_id,credibility) OR (user_idx,credibility)."
            )

    cred = np.clip(cred, 0.0, 1.0).astype(np.float32)
    p10, p50, p90 = np.percentile(cred, [10, 50, 90])
    print(f"[CRED] stats: min={cred.min():.4f} p10={p10:.4f} p50={p50:.4f} p90={p90:.4f} max={cred.max():.4f}")
    return cred


# =========================
# Eq (3.23) & (3.24) sparse operators
# =========================
def build_cred_weighted_mats(train_edges, num_users, num_items, cred_u: np.ndarray, device: str):
    """
    Implements Eq (3.23) and Eq (3.24).

    Let deg_u = |N(u)|, deg_i = |N(i)|

    Eq (3.23): e_i^{k+1} = sum_{u in N(i)} (c_hat[u] / sqrt(deg_u * deg_i)) * e_u^k
      => M_ui[row=item, col=user] = c_hat[u] / sqrt(deg_u*deg_i)

    Eq (3.24): e_u^{k+1} = sum_{i in N(u)} (1 / sqrt(deg_u * deg_i)) * e_i^k
      => M_iu[row=user, col=item] = 1 / sqrt(deg_u*deg_i)
    """
    u = train_edges[0].astype(np.int64)
    i = train_edges[1].astype(np.int64)

    deg_u = np.bincount(u, minlength=num_users).astype(np.float32)
    deg_i = np.bincount(i, minlength=num_items).astype(np.float32)

    denom = np.sqrt(np.maximum(deg_u[u] * deg_i[i], 1e-12)).astype(np.float32)

    w_ui = (cred_u[u] / denom).astype(np.float32)   # Eq (3.23)
    w_iu = (1.0 / denom).astype(np.float32)         # Eq (3.24)

    idx_ui = torch.tensor(np.vstack([i, u]), dtype=torch.long, device=device)
    val_ui = torch.tensor(w_ui, dtype=torch.float32, device=device)
    M_ui = torch.sparse_coo_tensor(idx_ui, val_ui, size=(num_items, num_users)).coalesce()

    idx_iu = torch.tensor(np.vstack([u, i]), dtype=torch.long, device=device)
    val_iu = torch.tensor(w_iu, dtype=torch.float32, device=device)
    M_iu = torch.sparse_coo_tensor(idx_iu, val_iu, size=(num_users, num_items)).coalesce()

    return M_ui, M_iu, deg_i


# =========================
# Credibility-Weighted LightGCN
# =========================
class CredLightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_layers, M_ui: torch.Tensor, M_iu: torch.Tensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.M_ui = M_ui.coalesce()
        self.M_iu = M_iu.coalesce()

        # Eq (3.22): init embeddings e_u^(0), e_i^(0) in R^d
        self.user_emb = torch.nn.Embedding(num_users, emb_dim)
        self.item_emb = torch.nn.Embedding(num_items, emb_dim)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate_all_layers(self):
        """
        Returns lists [e_u^(0)..e_u^(K)] and [e_i^(0)..e_i^(K)]
        """
        e_u = self.user_emb.weight
        e_i = self.item_emb.weight
        us = [e_u]
        is_ = [e_i]

        for _ in range(self.num_layers):
            # Eq (3.23): item <- user (cred-weighted)
            e_i = torch.sparse.mm(self.M_ui, e_u)

            # Eq (3.24): user <- item (standard)
            e_u = torch.sparse.mm(self.M_iu, is_[-1])

            us.append(e_u)
            is_.append(e_i)

        return us, is_

    def final_embeddings(self):
        """
        Eq (3.25): average embeddings across layers (0..K)
        """
        us, is_ = self.propagate_all_layers()
        e_u = torch.stack(us, dim=0).mean(dim=0)
        e_i = torch.stack(is_, dim=0).mean(dim=0)
        return e_u, e_i

    def score(self, users: torch.Tensor, items: torch.Tensor, e_u: torch.Tensor, e_i: torch.Tensor):
        """
        Eq (3.26): y_hat(u,i) = e_u^T e_i
        """
        return (e_u[users] * e_i[items]).sum(dim=1)

    def l2_reg(self, users, pos_items, neg_items):
        """
        L_reg: standard L2 regularization on embeddings (batch ego)
        """
        eu = self.user_emb.weight[users]
        ep = self.item_emb.weight[pos_items]
        en = self.item_emb.weight[neg_items]
        return (eu.norm(2, dim=1).pow(2) + ep.norm(2, dim=1).pow(2) + en.norm(2, dim=1).pow(2)).mean()


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
def evaluate_sampled(model: CredLightGCN, train_csr, test_csr, num_items: int, device: str):
    indptr_tr, indices_tr = train_csr
    indptr_te, indices_te = test_csr

    e_u, e_i = model.final_embeddings()
    e_u = e_u.to(device)
    e_i = e_i.to(device)

    rng = np.random.default_rng(cfg.seed + 999)

    users = np.where((indptr_te[1:] - indptr_te[:-1]) > 0)[0]
    if len(users) == 0:
        raise RuntimeError("No users with test interactions.")

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

        uvec = e_u[int(u)].unsqueeze(0)
        ivec = e_i[cand]
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


# =========================
# TRAIN LOOP (Eq 3.27 + 3.28 included)
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

    # credibility (only apply to users that exist in this LightGCN mapping)
    cred_u = load_credibility_vector(num_users, user2idx)

    train_csr = edges_to_user_csr(train_edges, num_users)
    val_csr = edges_to_user_csr(val_edges, num_users)
    test_csr = edges_to_user_csr(test_edges, num_users)

    device = cfg.device
    print("Using device:", device)

    # Eq (3.23)/(3.24)
    M_ui, M_iu, deg_i = build_cred_weighted_mats(train_edges, num_users, num_items, cred_u, device=device)

    # Eq (3.27): pop(i) normalized popularity based on TRAIN interactions
    pop = (deg_i / max(float(deg_i.max()), 1.0)).astype(np.float32)
    pop_t = torch.tensor(pop, device=device, dtype=torch.float32)

    model = CredLightGCN(num_users, num_items, cfg.emb_dim, cfg.num_layers, M_ui, M_iu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rng = np.random.default_rng(cfg.seed)

    indptr_tr, _ = train_csr
    train_users = np.where((indptr_tr[1:] - indptr_tr[:-1]) > 0)[0]
    if len(train_users) == 0:
        raise RuntimeError("No train users with interactions.")

    best_val = -1.0
    best_path = out / "model" / "best_model_cred.pt"

    indptr, indices = train_csr

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(train_users)

        total_loss = 0.0
        steps = 0

        for start in range(0, len(train_users), cfg.batch_size):
            batch_users = train_users[start : start + cfg.batch_size]

            used_users = []
            pos_items = []
            neg_items = []

            for u in batch_users:
                p = sample_pos_item(indptr, indices, int(u), rng)
                if p is None:
                    continue
                n = sample_neg_item(indptr, indices, int(u), num_items, rng)
                used_users.append(u)
                pos_items.append(p)
                neg_items.append(n)

            if len(pos_items) == 0:
                continue

            users_t = torch.tensor(used_users, device=device, dtype=torch.long)
            pos_t = torch.tensor(pos_items, device=device, dtype=torch.long)
            neg_t = torch.tensor(neg_items, device=device, dtype=torch.long)

            # Eq (3.25): compute final embeddings
            e_u, e_i = model.final_embeddings()

            # --- BPR main ranking objective (needed to learn recommendations)
            pos_scores = model.score(users_t, pos_t, e_u, e_i)  # Eq (3.26)
            neg_scores = model.score(users_t, neg_t, e_u, e_i)  # Eq (3.26)
            loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

            # Eq (3.27): L_fair = sum pop(i) * y_hat(u,i)
            # minibatch approximation over observed positives
            loss_fair = (pop_t[pos_t] * pos_scores).mean()

            # L_reg: L2 regularization
            loss_reg = model.l2_reg(users_t, pos_t, neg_t)

            # Eq (3.28): L = lambda_fair*L_fair + lambda_reg*L_reg
            # plus BPR for training signal
            loss = loss_bpr + cfg.lambda_fair * loss_fair + cfg.lambda_reg * loss_reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch:03d} | loss={avg_loss:.6f}")

        if epoch % cfg.eval_every == 0:
            model.eval()
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

    # Load best and test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

    test_res = evaluate_sampled(model, train_csr, test_csr, num_items, device)

    print("\nTEST metrics:")
    for K in cfg.Ks:
        r = test_res[K]
        print(f"  K={K}: P={r['precision']:.4f} R={r['recall']:.4f} NDCG={r['ndcg']:.4f} ({r['mode']})")


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
