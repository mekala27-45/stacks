"""Weighted implicit ALS and item cosine, implemented against SciPy sparse."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def _least_squares(
    interactions: csr_matrix, other: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    factors = other.shape[1]
    result = np.zeros((interactions.shape[0], factors), dtype=np.float64)
    base = other.T @ other + regularization * np.eye(factors)
    for row in range(interactions.shape[0]):
        ids = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
        if ids.size:
            observed = other[ids]
            result[row] = np.linalg.solve(
                base + alpha * observed.T @ observed,
                (1 + alpha) * observed.sum(axis=0),
            )
    return result


@dataclass
class ModelBundle:
    matrix: csr_matrix
    popularity: np.ndarray
    cosine: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray

    def score(self, user: int) -> dict[str, np.ndarray]:
        history = self.matrix[user].indices
        cosine = self.cosine[history].sum(axis=0) if history.size else self.popularity.copy()
        als = self.item_factors @ self.user_factors[user] if history.size else self.popularity.copy()
        blend = 0.55 * _normalize(cosine) + 0.35 * _normalize(als) + 0.10 * _normalize(self.popularity)
        return {"popularity": self.popularity.copy(), "item_cosine": cosine, "als": als, "blend": blend}


def train_models(matrix: csr_matrix, factors: int = 16, iterations: int = 6, seed: int = 17) -> ModelBundle:
    matrix = matrix.astype(np.float64)
    matrix.data[:] = 1.0
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    cooccurrence = (matrix.T @ matrix).toarray()
    lengths = np.sqrt(popularity)
    denominator = lengths[:, None] * lengths[None, :]
    cosine = np.divide(cooccurrence, denominator, out=np.zeros_like(cooccurrence), where=denominator > 0)
    np.fill_diagonal(cosine, 0)
    rng = np.random.default_rng(seed)
    items = rng.normal(0, 0.05, size=(matrix.shape[1], factors))
    users = np.zeros((matrix.shape[0], factors))
    transpose = matrix.T.tocsr()
    for _ in range(iterations):
        users = _least_squares(matrix, items, 0.1, 20.0)
        items = _least_squares(transpose, users, 0.1, 20.0)
    return ModelBundle(matrix, popularity, cosine, users, items)
