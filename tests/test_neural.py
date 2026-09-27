from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from packages.retrieval.two_tower import TwoTower, encode
from packages.session.rnn import RecurrentModel, train_recurrent
from scripts.build_neural_evidence import histories_from


def tower_fixture() -> TwoTower:
    rng = np.random.default_rng(42)
    return TwoTower(
        rng.uniform(0.1, 0.8, (4, 3)),
        rng.uniform(0.1, 0.8, (4, 3)),
        rng.uniform(0.1, 0.5, (3, 3)),
        rng.uniform(0.1, 0.5, (3, 3)),
    )


def tower_accumulators(model: TwoTower) -> list[np.ndarray]:
    return [
        np.zeros_like(array)
        for array in (
            model.user_embeddings,
            model.item_embeddings,
            model.user_projection,
            model.item_projection,
        )
    ]


@pytest.mark.parametrize(
    "parameter,index,accumulator",
    [("user_projection", (0, 0), 2), ("item_projection", (1, 0), 3), ("user_embeddings", (0, 1), 0)],
)
def test_two_tower_gradient_matches_finite_difference(
    parameter: str, index: tuple[int, int], accumulator: int
):
    model = tower_fixture()
    ids = np.arange(4)
    epsilon = 1e-5
    plus, minus = deepcopy(model), deepcopy(model)
    getattr(plus, parameter)[index] += epsilon
    getattr(minus, parameter)[index] -= epsilon
    numerical = (
        plus.step(ids, ids, tower_accumulators(plus), 0) - minus.step(ids, ids, tower_accumulators(minus), 0)
    ) / (2 * epsilon)
    original = float(getattr(model, parameter)[index])
    accumulators = tower_accumulators(model)
    rate = 1e-4
    model.step(ids, ids, accumulators, rate)
    analytical = (
        (original - getattr(model, parameter)[index])
        * np.sqrt(accumulators[accumulator][index] + 1e-6)
        / rate
    )
    assert analytical == pytest.approx(numerical, rel=1e-3, abs=1e-6)


def test_two_tower_optimization_and_zero_encoder_are_finite():
    model = tower_fixture()
    ids = np.arange(4)
    accumulators = tower_accumulators(model)
    initial = model.step(ids, ids, accumulators, 0)
    for _ in range(100):
        model.step(ids, ids, accumulators, 0.035)
    assert model.step(ids, ids, accumulators, 0) < initial * 0.75
    zero = encode(np.zeros((2, 3)), np.zeros((3, 2)))[0]
    assert np.isfinite(zero).all()
    assert not zero.any()


def rnn_fixture() -> RecurrentModel:
    rng = np.random.default_rng(13)
    return RecurrentModel(
        rng.normal(0, 0.08, (8, 3)), rng.normal(0, 0.08, (8, 3)), 0.3 * np.eye(3), np.zeros(3)
    )


def rnn_accumulators(model: RecurrentModel) -> list[np.ndarray]:
    return [np.zeros_like(array) for array in (model.inputs, model.outputs, model.recurrent, model.bias)]


def test_recurrent_bptt_gradient_matches_finite_difference():
    model = rnn_fixture()
    batch = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]])
    plus, minus = deepcopy(model), deepcopy(model)
    epsilon = 1e-5
    plus.recurrent[0, 1] += epsilon
    minus.recurrent[0, 1] -= epsilon
    numerical = (
        plus.step(batch, rnn_accumulators(plus), 0) - minus.step(batch, rnn_accumulators(minus), 0)
    ) / (2 * epsilon)
    original = model.recurrent[0, 1]
    accumulators = rnn_accumulators(model)
    rate = 1e-4
    model.step(batch, accumulators, rate)
    analytical = (original - model.recurrent[0, 1]) * np.sqrt(accumulators[2][0, 1] + 1e-6) / rate
    assert analytical == pytest.approx(numerical, rel=1e-3, abs=1e-6)


def test_recurrent_optimization_and_empty_session_are_finite():
    model = rnn_fixture()
    batch = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]])
    accumulators = rnn_accumulators(model)
    before = model.step(batch, accumulators, 0)
    for _ in range(100):
        model.step(batch, accumulators, 0.035)
    assert model.step(batch, accumulators, 0) < before * 0.75
    assert not model.encode(np.array([], dtype=int)).any()
    assert np.isfinite(model.encode(batch[0])).all()
    with pytest.raises(ValueError, match="histories"):
        train_recurrent({0: np.arange(3)}, 8, examples=8)


def test_neural_histories_exclude_holdout_and_preserve_source_order():
    frame = pd.DataFrame(
        {
            "user_id": [1] * 10,
            "book_id": [3, 2, 1, 5, 4, 7, 6, 9, 8, 10],
            "rating": [5] * 10,
            "source_row": np.arange(10),
        }
    )
    mask = (frame.source_row < 7).to_numpy()
    before = histories_from(frame, mask, {1: 0})
    frame.loc[~mask, "rating"] = 1
    frame.loc[~mask, "book_id"] = 99
    after = histories_from(frame, mask, {1: 0})
    np.testing.assert_array_equal(before[0], [2, 1, 0, 4, 3, 6, 5])
    np.testing.assert_array_equal(before[0], after[0])
