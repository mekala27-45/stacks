"""Train-only collaborative matrix and held-out relevance construction."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def matrices(
    ratings: pd.DataFrame,
    train_mask: np.ndarray,
    user_map: dict[int, int],
    catalog_size: int = 1000,
) -> tuple[csr_matrix, dict[int, set[int]], dict[int, set[int]]]:
    train = ratings[train_mask]
    positive = train[train.rating >= 4]
    matrix = csr_matrix(
        (np.ones(len(positive)), (positive.user_id.map(user_map), positive.book_id - 1)),
        shape=(len(user_map), catalog_size),
    )
    seen = {
        user_map[int(uid)]: {int(item) - 1 for item in frame.book_id}
        for uid, frame in train.groupby("user_id")
    }
    test = ratings[~train_mask]
    truth = {
        user_map[int(uid)]: {int(item) - 1 for item in frame[frame.rating >= 4].book_id}
        - seen.get(user_map[int(uid)], set())
        for uid, frame in test.groupby("user_id")
    }
    return matrix, seen, {uid: items for uid, items in truth.items() if items}
