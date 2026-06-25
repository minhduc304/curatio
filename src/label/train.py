"""C9 — train + export the quality classifier. See TDD §4.4, FR9.

score>=4 -> 1, <=2 -> 0, drop 3s; train sklearn LogisticRegression on Embed v4
vectors (raw, no scaler — inference is a plain dot product, mirrored in Rust);
export coeffs to models/quality_model.json. No API cost: embeddings come from the
warm cache that label.judge's docs populated.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from src.config import settings
from src.embedding.service import EmbeddingService
from src.models import CurationConfig, QualityModel

THRESHOLD = 0.5  # default decision cutoff; consumer uses CurationConfig.quality_threshold


def _load_labels() -> list[dict]:
    with open(settings.cache_dir / "label_cache.json") as f:
        return json.load(f)


def run() -> None:
    config = CurationConfig()
    records = _load_labels()

    texts: list[str] = []
    y: list[int] = []
    for r in records:
        score = r["edu_score"]
        if score >= 4:
            label = 1
        elif score <= 2:
            label = 0
        else:
            continue  # drop ambiguous 3s
        texts.append(r["text"])
        y.append(label)

    y_arr = np.asarray(y)
    n_pos, n_neg = int(y_arr.sum()), int((y_arr == 0).sum())
    print(f"{len(records)} labels -> {len(y)} training rows (pos={n_pos}, neg={n_neg})")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("need both positive and negative examples to train; relabel more docs")

    embedder = EmbeddingService(config.embed_dim, config.embed_batch_size)
    X = embedder.embed(texts)  # cache-aware; raw L2-normalized Embed v4 vectors

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y_arr)

    auc = roc_auc_score(y_arr, clf.predict_proba(X)[:, 1])
    print(f"train-set AUC={auc:.3f}, accuracy={clf.score(X, y_arr):.3f}")
    if min(n_pos, n_neg) >= 5:
        cv = cross_val_score(clf, X, y_arr, cv=5, scoring="roc_auc")
        print(f"5-fold CV AUC={cv.mean():.3f} (+/-{cv.std():.3f})")

    model = QualityModel(
        coef=clf.coef_[0].astype(float).tolist(),
        intercept=float(clf.intercept_[0]),
        embed_dim=config.embed_dim,
        threshold=THRESHOLD,
    )
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "w") as f:
        f.write(model.model_dump_json(indent=2))
    print(f"exported {settings.model_path} (embed_dim={config.embed_dim})")


if __name__ == "__main__":
    run()
