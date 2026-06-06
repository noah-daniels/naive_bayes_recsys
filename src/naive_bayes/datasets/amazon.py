import json
import pathlib
import gzip
import pandas as pd

from naive_bayes.datasets.utils import BASE_PATH_RAW, filter_interactions, process_interactions_df


def parse_amazon_raw(path: pathlib.Path):
    def parse():
        g = gzip.open(path, "rb")
        for l in g:
            yield json.loads(l)
        # if path.suffix == "gz":
        # else:
        #     with open(path) as file:
        #         for l in file:
        #             yield eval(l)

    df = []
    for d in parse():
        df.append(d)
    return pd.DataFrame(df)


def process_amazon_raw(path, min_items_per_user, min_users_per_item, min_rating):
    USER_IDX = "reviewerID"
    ITEM_IDX = "asin"
    df_ratings = parse_amazon_raw(path)
    df_interactions = df_ratings[df_ratings["overall"] >= min_rating].drop_duplicates(subset=[USER_IDX, ITEM_IDX])
    df_interactions = filter_interactions(df_interactions, min_items_per_user,
                                          min_users_per_item, user_idx=USER_IDX, item_idx=ITEM_IDX)
    df_interactions, _, _ = process_interactions_df(df_interactions.reset_index(), user_idx=USER_IDX, item_idx=ITEM_IDX)

    return df_interactions


def load_amazon_generic(name, min_items_per_user=5, min_users_per_item=5, min_rating=4, base_path: pathlib.Path | None = None):
    # default path
    if base_path is None:
        base_path = BASE_PATH_RAW / "amazon"

    # process raw file if preprocessed version doesn't exist yet
    processed_path = base_path / f"{name}.parquet"
    if not processed_path.is_file():
        df = process_amazon_raw(base_path / f"{name}.json.gz", min_items_per_user, min_users_per_item, min_rating)
        df.to_parquet(processed_path)

    # read preprocessed version
    return pd.read_parquet(processed_path)


def process_amazon_ratings_raw(path, min_items_per_user, min_users_per_item, min_rating):
    df_ratings = pd.read_csv(path, names=["user_id", "item_id", "rating", "timestamp"])
    df_interactions = df_ratings[df_ratings["rating"] >= min_rating].drop_duplicates(subset=["user_id", "item_id"])
    df_interactions = filter_interactions(df_interactions, min_items_per_user, min_users_per_item)
    df_interactions, _, _ = process_interactions_df(df_interactions.reset_index())

    return df_interactions


def load_amazon_ratings_generic(name, min_items_per_user=5, min_users_per_item=5, min_rating=4, base_path: pathlib.Path | None = None):
    # default path
    if base_path is None:
        base_path = BASE_PATH_RAW / "amazon"

    processed_path = base_path / f"{name}.parquet"
    if not processed_path.is_file():
        df = process_amazon_ratings_raw(base_path / f"{name}.csv", min_items_per_user, min_users_per_item, min_rating)
        df.to_parquet(processed_path)

    # read preprocessed version
    return pd.read_parquet(processed_path)


def load_amazon_books(base_path: pathlib.Path | None = None):
    return load_amazon_ratings_generic("ratings_Books", min_items_per_user=5, min_users_per_item=100,  min_rating=4, base_path=base_path)
