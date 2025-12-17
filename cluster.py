import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from openpyxl import Workbook

# -----------------------
# Load embeddings
# -----------------------
X = np.load("embeddings.npy")

# -----------------------
# Load metadata (aligné)
# -----------------------
meta = []
with open("embeddings_meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "error" not in obj:
            meta.append(obj)

assert len(meta) == len(X), f"Mismatch meta={len(meta)} embeddings={len(X)}"

# -----------------------
# Clustering
# -----------------------
X2 = PCA(n_components=50, random_state=0).fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
labels = kmeans.fit_predict(X2)

# -----------------------
# Group by cluster
# -----------------------
clusters = defaultdict(list)
for label, m in zip(labels, meta):
    clusters[label].append(m)

# -----------------------
# Write Excel (1 sheet / cluster)
# -----------------------
wb = Workbook()
wb.remove(wb.active)  # supprime la feuille par défaut

for cluster_id in sorted(clusters):
    ws = wb.create_sheet(title=f"Cluster_{cluster_id}")

    # Header
    ws.append(["Artist", "Title", "Album", "Deezer link"])

    # Rows
    for m in clusters[cluster_id]:
        ws.append([
            m.get("artist"),
            m.get("title"),
            m.get("album"),
            m.get("link")
        ])

# Save
output_path = "clusters.xlsx"
wb.save(output_path)

print(f"✔ Fichier Excel créé : {output_path}")
