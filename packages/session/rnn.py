"""Small Elman recurrent model trained by BPTT and in-batch next-item loss."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RecurrentModel:
    inputs: np.ndarray
    outputs: np.ndarray
    recurrent: np.ndarray
    bias: np.ndarray
    temperature: float = 0.3

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        hidden = np.zeros(self.inputs.shape[1], dtype=np.float32)
        for item in sequence[-12:]:
            hidden = np.tanh(self.inputs[item] + hidden @ self.recurrent + self.bias)
        return hidden

    def step(self, batch: np.ndarray, accumulators: list[np.ndarray], rate: float) -> float:
        batch_size, length = batch.shape
        hidden = [np.zeros((batch_size, self.inputs.shape[1]), dtype=np.float32)]
        losses, local_gradients, output_gradients = [], [], []
        for time in range(length - 1):
            state = np.tanh(self.inputs[batch[:, time]] + hidden[-1] @ self.recurrent + self.bias)
            hidden.append(state)
            targets = self.outputs[batch[:, time + 1]]
            logits = state @ targets.T / self.temperature
            duplicate = batch[:, time + 1, None] == batch[None, :, time + 1]
            np.fill_diagonal(duplicate, False)
            logits[duplicate] = -1e9
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            losses.append(float(-np.log(np.maximum(np.diag(probabilities), 1e-12)).mean()))
            probabilities[np.arange(batch_size), np.arange(batch_size)] -= 1
            probabilities /= batch_size * (length - 1) * self.temperature
            local_gradients.append(probabilities @ targets)
            output_gradients.append(probabilities.T @ state)
        recurrent_gradient = np.zeros_like(self.recurrent)
        bias_gradient = np.zeros_like(self.bias)
        future = np.zeros_like(hidden[0])
        input_gradients = []
        for time in reversed(range(length - 1)):
            gradient = (local_gradients[time] + future) * (1 - hidden[time + 1] ** 2)
            gradient = np.clip(gradient, -1, 1)
            recurrent_gradient += hidden[time].T @ gradient
            bias_gradient += gradient.sum(axis=0)
            input_gradients.append((time, gradient))
            future = gradient @ self.recurrent.T
        if rate:
            for time, gradient in input_gradients:
                indices = batch[:, time]
                np.add.at(accumulators[0], indices, gradient * gradient)
                np.add.at(self.inputs, indices, -rate * gradient / np.sqrt(accumulators[0][indices] + 1e-6))
            for time, gradient in enumerate(output_gradients):
                indices = batch[:, time + 1]
                np.add.at(accumulators[1], indices, gradient * gradient)
                np.add.at(self.outputs, indices, -rate * gradient / np.sqrt(accumulators[1][indices] + 1e-6))
            for parameter, gradient, accumulator in zip(
                (self.recurrent, self.bias),
                (recurrent_gradient, bias_gradient),
                accumulators[2:],
                strict=True,
            ):
                accumulator += gradient * gradient
                parameter -= rate * gradient / np.sqrt(accumulator + 1e-6)
        return float(np.mean(losses))


def train_recurrent(
    histories: dict[int, np.ndarray], catalog_size: int, examples: int = 24000, seed: int = 413
) -> tuple[RecurrentModel, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    eligible = [history for history in histories.values() if len(history) >= 7]
    if not eligible:
        raise ValueError("Recurrent training requires histories of at least7 positives")
    sequences = []
    for _ in range(examples):
        sequence = eligible[int(rng.integers(len(eligible)))]
        start = int(rng.integers(0, len(sequence) - 6))
        sequences.append(sequence[start : start + 7])
    batches = np.array(sequences, dtype=np.int32)
    model = RecurrentModel(
        rng.normal(0, 0.08, (catalog_size, 16)).astype(np.float32),
        rng.normal(0, 0.08, (catalog_size, 16)).astype(np.float32),
        (0.3 * np.eye(16)).astype(np.float32),
        np.zeros(16, dtype=np.float32),
    )
    accumulators = [
        np.zeros_like(array) for array in (model.inputs, model.outputs, model.recurrent, model.bias)
    ]
    before = model.step(batches[:64], accumulators, 0)
    losses = [
        model.step(batches[start : start + 64], accumulators, 0.035) for start in range(0, examples, 64)
    ]
    after = model.step(batches[:64], accumulators, 0)
    return model, {
        "status": "measured",
        "architecture": "16-dimensional Elman tanh recurrent encoder with learned input/output item embeddings",
        "objective": "in-batch next-item softmax, backpropagation through6 time steps",
        "training_sequences": examples,
        "sequence_length": 7,
        "batch_size": 64,
        "seed": seed,
        "learning_rate": 0.035,
        "optimizer": "AdaGrad with clipped recurrent gradients",
        "fixed_training_batch_loss_before": before,
        "fixed_training_batch_loss_after": after,
        "mean_training_loss": float(np.mean(losses)),
        "detail": "A small Elman RNN, not a GRU or pretrained recurrent model. Sequences preserve source-file order, an unaudited chronology proxy. Final recommendations condition on the last12 positive training interactions; all training targets precede the final cutoff.",
    }
