"""Two independent nonlinear ID encoders trained with in-batch negatives.

Each tower is an embedding lookup followed by a learned projection, ReLU, and
L2 normalization. Gradients are explicit NumPy; no implicit-MF objective is used.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


def encode(embeddings: np.ndarray, projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.maximum(embeddings @ projection, 0)
    norm = np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8)
    return hidden / norm, hidden, norm


def encoder_gradient(
    gradient: np.ndarray, output: np.ndarray, hidden: np.ndarray, norm: np.ndarray
) -> np.ndarray:
    return np.asarray(
        (gradient - output * np.sum(gradient * output, axis=1, keepdims=True)) / norm * (hidden > 0)
    )


@dataclass
class TwoTower:
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    user_projection: np.ndarray
    item_projection: np.ndarray
    temperature: float = 0.15

    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        return encode(self.user_embeddings, self.user_projection)[0], encode(
            self.item_embeddings, self.item_projection
        )[0]

    def step(
        self, users: np.ndarray, items: np.ndarray, accumulators: list[np.ndarray], rate: float
    ) -> float:
        user_input = self.user_embeddings[users].copy()
        item_input = self.item_embeddings[items].copy()
        user, user_hidden, user_norm = encode(user_input, self.user_projection)
        item, item_hidden, item_norm = encode(item_input, self.item_projection)
        logits = user @ item.T / self.temperature
        duplicate = items[:, None] == items[None, :]
        np.fill_diagonal(duplicate, False)
        logits[duplicate] = -1e9
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(np.diag(probabilities), 1e-12)).mean())
        derivative = probabilities.copy()
        derivative[np.arange(len(users)), np.arange(len(users))] -= 1
        derivative /= len(users) * self.temperature
        user_gradient = encoder_gradient(derivative @ item, user, user_hidden, user_norm)
        item_gradient = encoder_gradient(derivative.T @ user, item, item_hidden, item_norm)
        user_lookup_gradient = user_gradient @ self.user_projection.T
        item_lookup_gradient = item_gradient @ self.item_projection.T
        projection_gradients = [user_input.T @ user_gradient, item_input.T @ item_gradient]
        if rate:
            for lookup, indices, gradient, accumulator in (
                (self.user_embeddings, users, user_lookup_gradient, accumulators[0]),
                (self.item_embeddings, items, item_lookup_gradient, accumulators[1]),
            ):
                np.add.at(accumulator, indices, gradient * gradient)
                np.add.at(lookup, indices, -rate * gradient / np.sqrt(accumulator[indices] + 1e-6))
            for projection, gradient, accumulator in zip(
                (self.user_projection, self.item_projection),
                projection_gradients,
                accumulators[2:],
                strict=True,
            ):
                accumulator += gradient * gradient
                projection -= rate * gradient / np.sqrt(accumulator + 1e-6)
        return loss


def train_two_tower(
    matrix: csr_matrix, pairs: int = 400000, seed: int = 311
) -> tuple[TwoTower, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    user_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), np.diff(matrix.indptr))
    chosen = rng.choice(matrix.nnz, size=min(pairs, matrix.nnz), replace=False)
    users, items = user_ids[chosen], matrix.indices[chosen]
    model = TwoTower(
        rng.normal(0, 0.08, (matrix.shape[0], 32)).astype(np.float32),
        rng.normal(0, 0.08, (matrix.shape[1], 32)).astype(np.float32),
        rng.normal(0, 0.2, (32, 16)).astype(np.float32),
        rng.normal(0, 0.2, (32, 16)).astype(np.float32),
    )
    accumulators = [
        np.zeros_like(array)
        for array in (
            model.user_embeddings,
            model.item_embeddings,
            model.user_projection,
            model.item_projection,
        )
    ]
    diagnostic_users, diagnostic_items = users[:128], items[:128]
    before = model.step(diagnostic_users, diagnostic_items, accumulators, 0)
    losses = []
    for start in range(0, len(users), 128):
        stop = min(start + 128, len(users))
        if stop - start > 1:
            losses.append(model.step(users[start:stop], items[start:stop], accumulators, 0.035))
    after = model.step(diagnostic_users, diagnostic_items, accumulators, 0)
    return model, {
        "status": "measured",
        "architecture": "Independent user-ID and item-ID embeddings32 -> learned projection16 -> ReLU -> L2 normalization",
        "objective": "in-batch softmax cross entropy",
        "training_pairs": len(users),
        "batch_size": 128,
        "temperature": model.temperature,
        "learning_rate": 0.035,
        "optimizer": "AdaGrad",
        "seed": seed,
        "fixed_training_batch_loss_before": before,
        "fixed_training_batch_loss_after": after,
        "mean_training_loss": float(np.mean(losses)),
        "detail": "All sampled positives precede the final source-order cutoff. Duplicate item IDs within a batch are masked as false negatives. ID encoders do not solve cold-user or cold-item generalization. Optimization loss is a training diagnostic, not held-out evidence.",
    }
