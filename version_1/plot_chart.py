import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# ONLY CHANGE: robust loading paths
# =========================
FEATURE_FILE = os.path.join(BASE_DIR, "dataset", "user_features_new.csv")
LABEL_FILE   = os.path.join(BASE_DIR, "dataset", "user_labels.csv")

FALLBACK_DATASET_DIR = "/home/dsai-st125024/project/Beyond-Binary-Fake-User-Detection-A-Credibility-Aware-Graph-based-Recommender-System/dataset"

if not os.path.exists(FEATURE_FILE):
    FEATURE_FILE = os.path.join(FALLBACK_DATASET_DIR, "user_features.csv")
if not os.path.exists(LABEL_FILE):
    LABEL_FILE = os.path.join(FALLBACK_DATASET_DIR, "user_labels.csv")

# =========================
# KEEP EVERYTHING SAME BELOW
# =========================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------- load
features_df = pd.read_csv(FEATURE_FILE)
labels_df   = pd.read_csv(LABEL_FILE)

# strip spaces in headers
features_df.columns = features_df.columns.str.strip()
labels_df.columns   = labels_df.columns.str.strip()

# optional: standardize feature column casing for lookup only (keep original columns)
features_lower_map = {c.lower().strip(): c for c in features_df.columns}

# ✅ Ensure the new features exist (RNR, ETG)
for required in ["rnr", "etg"]:
    if required not in features_lower_map:
        print("features_df columns:", list(features_df.columns))
        raise SystemExit(f"❌ Required feature '{required.upper()}' not found in user_features.csv")

# If the columns exist but are strings, force numeric
for col_key in ["rnr", "etg"]:
    col = features_lower_map[col_key]
    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

# find user_id column robustly
def find_col(df, candidates):
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in norm:
            return norm[k]
    return None

user_col_f = find_col(features_df, ["user_id", "userid", "user"])
user_col_l = find_col(labels_df,   ["user_id", "userid", "user"])
if user_col_f is None or user_col_l is None:
    print("features_df columns:", list(features_df.columns))
    print("labels_df columns  :", list(labels_df.columns))
    raise SystemExit("❌ user_id column not found in one of the files.")

# ---- merge ALL label columns from labels_df (except user_id)
label_candidates = [c for c in labels_df.columns if c != user_col_l]

df = pd.merge(
    features_df,
    labels_df[[user_col_l] + label_candidates],
    left_on=user_col_f,
    right_on=user_col_l,
    how="inner"
)

# print merged columns so you can see what happened
print("✅ Columns after merge:\n", list(df.columns))

# ---- detect label column in merged df
possible = [
    "label", "label_x", "label_y",
    "is_fake", "is_fake_x", "is_fake_y",
    "fake", "fake_x", "fake_y",
    "class", "class_x", "class_y",
    "target", "target_x", "target_y",
    "y", "y_x", "y_y"
]

label_col = None
norm_map = {c.lower().strip(): c for c in df.columns}

for p in possible:
    if p in norm_map:
        label_col = norm_map[p]
        break

if label_col is None:
    brought = [c for c in label_candidates if c in df.columns]
    if len(brought) == 0:
        raise SystemExit("❌ No label-like column found after merge.")
    label_col = brought[-1]
    print("⚠️ Label column not found by name; using fallback:", label_col)
else:
    print("✅ Detected label column:", label_col)

# standardize column names
df = df.rename(columns={user_col_f: "user_id", label_col: "label"})

# IMPORTANT: if user_id exists twice, drop the duplicate right-side one
dup_user_cols = [c for c in df.columns if c.lower().strip() in ["user_id", "userid"]]
for c in dup_user_cols:
    if c != "user_id":
        df = df.drop(columns=[c])

# ---- normalize label values
if pd.api.types.is_numeric_dtype(df["label"]):
    df["label"] = df["label"].apply(lambda x: "fake" if int(x) == 1 else "genuine")
else:
    df["label"] = df["label"].astype(str).str.strip().str.lower()

df = df[df["label"].isin(["fake", "genuine"])].copy()

# ---- numeric feature columns
exclude = {"user_id", "label"}
features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# ✅ Make sure RNR + ETG are included (even if dtype got weird somewhere)
# (Already forced numeric above, but this guarantees they’re in the plot list.)
for must in ["RNR", "ETG"]:
    if must in df.columns and must not in features:
        if pd.api.types.is_numeric_dtype(df[must]):
            features.append(must)

print("Detected numeric features:", features)

# ---- plot density for each feature
print("\nGenerating distribution plots...")
for feature in features:
    plt.figure(figsize=(8, 5))

    genuine_vals = df[df["label"] == "genuine"][feature].dropna()
    fake_vals    = df[df["label"] == "fake"][feature].dropna()

    # avoid seaborn crash when a feature is constant or too few values
    if genuine_vals.nunique() < 2 or fake_vals.nunique() < 2:
        print(f"⚠️ Skipping {feature} (not enough variance for KDE)")
        plt.close()
        continue

    sns.kdeplot(genuine_vals, label="Genuine", fill=True, alpha=0.4)
    sns.kdeplot(fake_vals,    label="Fake",    fill=True, alpha=0.4)

    plt.title(f"Distribution of {feature} (Fake vs Genuine)")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_FOLDER, f"distribution_{feature}.png"), dpi=300)
    plt.close()

print("\n✅ Distribution plots saved in:", OUTPUT_FOLDER)

# Helpful debug print (optional)
print("✅ Loaded FEATURES from:", FEATURE_FILE)
print("✅ Loaded LABELS from   :", LABEL_FILE)