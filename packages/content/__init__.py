"""Frozen snapshot title, author and reader-tag TF-IDF cold-item retrieval."""

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectors(catalog: list[dict[str, Any]]) -> csr_matrix:
    documents = [
        f"{item['title']} {item['author']} {' '.join(item['tags'])} {item['genre']}" for item in catalog
    ]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=40000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return csr_matrix(vectorizer.fit_transform(documents), dtype=np.float32)
