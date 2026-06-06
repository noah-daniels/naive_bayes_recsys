import numpy as np
import scipy as sp
import pandas as pd
import pathlib

import dotenv
import os

dotenv.load_dotenv()
BASE_PATH_RAW = pathlib.Path(
    os.environ.get("RAW_DATA_PATH", pathlib.Path(__file__).parent.parent.parent.parent / "data/raw")
)


def filter_interactions(df, min_items_per_user, min_users_per_item, user_idx="user_id", item_idx="item_id"):
    """
    Ensure:
    - each user has interacted with at least `min_items_per_user`
    - each item was interacted with by at least `min_users_per_item`
    """
    user_frequency = df[user_idx].value_counts()
    item_frequency = df[item_idx].value_counts()
    while (user_frequency < min_items_per_user).sum() > 0 or (item_frequency < min_users_per_item).sum() > 0:
        frequent_users = user_frequency[user_frequency >= min_items_per_user].index
        frequent_items = item_frequency[item_frequency >= min_users_per_item].index
        df = df[df[user_idx].isin(frequent_users) & df[item_idx].isin(frequent_items)]
        user_frequency = df[user_idx].value_counts()
        item_frequency = df[item_idx].value_counts()

    return df


def factorize_series_sorted(series: pd.Series) -> pd.Series:
    """
    Determine mapping to makes values contiguous and sorted by occurrence count.
    """
    sorted_original_ids = series.value_counts(sort=True, ascending=False).index
    new_ids = np.arange(len(sorted_original_ids))
    mapper = pd.Series(new_ids, index=sorted_original_ids, name="id")
    mapper.rename_axis(f"original_id", inplace=True)
    mapper.sort_index(inplace=True)
    return mapper


def process_interactions_df(df, user_idx="user_id", item_idx="item_id"):
    """
    Map users and items to new id space (sorted and contiguous) and remove all other columns.
    """
    NEW_USER_IDX = "user_id"
    NEW_ITEM_IDX = "item_id"

    # sort and factorize users
    user_mapper = factorize_series_sorted(df[user_idx])
    user_ids = df.merge(user_mapper, left_on=user_idx, right_index=True)["id"]

    # sort and factorize items
    item_mapper = factorize_series_sorted(df[item_idx])
    item_ids = df.merge(item_mapper, left_on=item_idx, right_index=True)["id"]

    # create new dataframe in new id space
    result = pd.DataFrame({NEW_USER_IDX: user_ids, NEW_ITEM_IDX: item_ids})

    # sanity check id factorization
    assert result[NEW_USER_IDX].max() + 1 == result[NEW_USER_IDX].nunique()
    assert result[NEW_ITEM_IDX].max() + 1 == result[NEW_ITEM_IDX].nunique()

    return result, user_mapper, item_mapper


def create_interaction_array(df: pd.DataFrame, user_idx="user_id", item_idx="item_id") -> sp.sparse.csr_array:
    """
    Creates a sparse binary interaction matrix from a dataframe of interaction.
    Dataframe must contain two columns indicating the user and items.
    """
    values = np.ones(df.shape[0])
    indices = df[[user_idx, item_idx]].values
    indices = indices[:, 0], indices[:, 1]

    shape = (indices[0].max() + 1, indices[1].max() + 1)

    matrix = sp.sparse.csr_array((values, indices), shape=shape, dtype=np.int32)
    return matrix
