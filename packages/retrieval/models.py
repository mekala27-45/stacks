"""Train-only sparse item cosine, weighted implicit ALS, and shared inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, load_npz


def _normalize(values: np.ndarray) -> np.ndarray:
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def _least_squares(
    interactions: csr_matrix, other: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    factors = other.shape[1]
    result = np.zeros((interactions.shape[0], factors), dtype=np.float32)
    base = other.T @ other + regularization * np.eye(factors, dtype=np.float32)
    for row in range(interactions.shape[0]):
        ids = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
        if ids.size:
            observed = other[ids]
            result[row] = np.linalg.solve(
                base + alpha * observed.T @ observed, (1 + alpha) * observed.sum(axis=0)
            )
    return result


def sparse_item_cosine(matrix: csr_matrix, neighbors: int = 100) -> csr_matrix:
    """Block-exact cooccurrence, retaining each item's strongest neighbors.

    Pruning is part of the model definition. Full-ranking evaluation still scores
    every catalog item, and the same pruned matrix is served online.
    """
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    norms = np.sqrt(counts)
    n = matrix.shape[1]
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    csc = matrix.tocsc()
    for start in range(0, n, 100):
        end = min(start + 100, n)
        block = (csc[:, start:end].T @ matrix).toarray().astype(np.float32)
        denominator = norms[start:end, None] * norms[None, :]
        np.divide(block, denominator, out=block, where=denominator > 0)
        block[denominator == 0] = 0
        for offset, values in enumerate(block):
            values[start + offset] = 0
            nonzero = np.flatnonzero(values > 0)
            chosen = nonzero[np.lexsort((nonzero, -values[nonzero]))[:neighbors]]
            chosen.sort()
            indices.extend(chosen.tolist())
            data.extend(values[chosen].tolist())
            indptr.append(len(data))
    return csr_matrix(
        (
            np.array(data, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        ),
        shape=(n, n),
    )


@dataclass
class ModelBundle:
    matrix: csr_matrix
    popularity: np.ndarray
    cosine: csr_matrix
    user_factors: np.ndarray
    item_factors: np.ndarray
    content_vectors: csr_matrix | None = None
    ranker: Any = None
    user_ids: np.ndarray | None = None
    catalog_ids: np.ndarray | None = None
    seen: csr_matrix | None = None

    def fold_in(self, history: np.ndarray) -> np.ndarray:
        if history.size == 0:
            return np.zeros(self.item_factors.shape[1], dtype=np.float32)
        observed = self.item_factors[np.unique(history)]
        base = self.item_factors.T @ self.item_factors + 0.1 * np.eye(
            self.item_factors.shape[1], dtype=np.float32
        )
        return np.linalg.solve(base + 20 * observed.T @ observed, 21 * observed.sum(axis=0))

    def score_history(
        self, history: np.ndarray, user_factor: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        history = np.asarray(history, dtype=int)
        cosine = (
            np.asarray(self.cosine[history].sum(axis=0)).ravel() if history.size else self.popularity.copy()
        )
        factor = self.fold_in(history) if user_factor is None else user_factor
        als = self.item_factors @ factor if history.size else self.popularity.copy()
        blend = 0.55 * _normalize(cosine) + 0.35 * _normalize(als) + 0.10 * _normalize(self.popularity)
        scores = {"popularity": self.popularity.copy(), "item_cosine": cosine, "als": als, "blend": blend}
        if self.content_vectors is not None:
            if history.size:
                profile = csr_matrix(self.content_vectors[history].sum(axis=0))
                norm = float(np.sqrt(profile.multiply(profile).sum()))
                content = (self.content_vectors @ (profile / max(norm, 1e-12)).T).toarray().ravel()
            else:
                content = self.popularity.copy()
            scores["content"] = content
        if self.ranker is not None:
            from packages.ranking.learned import ranker_features

            scores["lambdamart"] = self.ranker.predict(ranker_features(scores, history.size), num_threads=1)
        return scores

    def score(self, user: int) -> dict[str, np.ndarray]:
        return self.score_history(self.matrix[user].indices, self.user_factors[user])


def train_models(
    matrix: csr_matrix,
    factors: int = 16,
    iterations: int = 6,
    seed: int = 17,
    content_vectors: csr_matrix | None = None,
) -> ModelBundle:
    matrix = matrix.astype(np.float32)
    matrix.data[:] = 1.0
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    cosine = sparse_item_cosine(matrix)
    rng = np.random.default_rng(seed)
    items = rng.normal(0, 0.05, size=(matrix.shape[1], factors)).astype(np.float32)
    users = np.zeros((matrix.shape[0], factors), dtype=np.float32)
    transpose = matrix.T.tocsr()
    for _ in range(iterations):
        users = _least_squares(matrix, items, 0.1, 20.0)
        items = _least_squares(transpose, users, 0.1, 20.0)
    return ModelBundle(matrix, popularity, cosine, users, items, content_vectors)


def load_bundle(results: Path) -> ModelBundle:
    artifacts = np.load(results / "model-artifacts.npz")
    ranker = None
    if (results / "lambdamart.txt").exists():
        import lightgbm as lgb

        ranker = lgb.Booster(model_file=str(results / "lambdamart.txt"))
    return ModelBundle(
        csr_matrix(load_npz(results / "train-positive.npz")),
        artifacts["popularity"],
        csr_matrix(load_npz(results / "cosine.npz")),
        artifacts["user_factors"],
        artifacts["item_factors"],
        csr_matrix(load_npz(results / "content-vectors.npz")),
        ranker,
        artifacts["user_ids"],
        artifacts["catalog_ids"],
        csr_matrix(load_npz(results / "train-seen.npz")),
    )
