
import pickle
import numpy as np
import pandas as pd

OUT_DIR = "dataset/graph_pyg_parent_asin"

cred = np.load(f"{OUT_DIR}/credibility_scores.npy")

with open(f"{OUT_DIR}/user2idx.pkl", "rb") as f:
    user2idx = pickle.load(f)

# invert mapping
idx2user = {idx: uid for uid, idx in user2idx.items()}

df = pd.DataFrame({
    "user_idx": np.arange(len(cred)),
    "user_id": [idx2user.get(i, None) for i in range(len(cred))],
    "credibility": cred
})

df.to_csv(f"{OUT_DIR}/credibility_scores_with_user_id.csv", index=False)
print("Saved:", f"{OUT_DIR}/credibility_scores_with_user_id.csv")
print(df.head())

