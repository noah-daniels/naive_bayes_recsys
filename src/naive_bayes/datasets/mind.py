import pandas as pd
import pathlib
from naive_bayes.datasets.utils import BASE_PATH_RAW, create_interaction_array, filter_interactions, factorize_series_sorted


def parse_mind_raw(raw_path: pathlib.Path, processed_path: pathlib.Path):
    if not processed_path.is_file():
        # read file
        df = pd.read_csv(
            raw_path, sep="\t", names=["id", "user", "timestamp", "history", "impressions"]
        )

        # sort by time
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %I:%M:%S %p")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.drop(columns=["id"], inplace=True)

        # create lists
        df["history"] = df["history"].apply(
            lambda x: x.split(" ") if isinstance(x, str) else []
        )
        impressions = df["impressions"].apply(
            lambda x: [
                (s[:-2], int(s[-1]))
                for s in (x.split(" ") if isinstance(x, str) else [])
            ]
        )
        df["clicks"] = impressions.apply(lambda x: [item for item, click in x if click])
        df["impressions"] = impressions.apply(lambda x: [item for item, _ in x])
        df.to_parquet(processed_path)

    return pd.read_parquet(processed_path)


def load_mind(base_path: pathlib.Path = None):
    if base_path is None:
        base_path = BASE_PATH_RAW / "mind"

    h5_path = base_path / "mind.h5"

    if not h5_path.is_file():
        # load raw dataset
        # we combine training and validation data as we are handling the split ourselves later on
        df = pd.concat((
            parse_mind_raw(base_path / "MINDlarge_dev" / "behaviors.tsv", base_path / "behaviors_dev.parquet"),
            parse_mind_raw(base_path / "MINDlarge_train" / "behaviors.tsv", base_path / "behaviors_train.parquet")
        )).sort_values("timestamp").reset_index(drop=True)

        # extract click and impression interactions
        # remove duplicates
        df_clicks = pd.merge(df.clicks.explode(), df[["user", "timestamp"]], left_index=True, right_index=True) \
            .rename(columns={"user": "user_id", "clicks": "item_id"}) \
            .drop_duplicates(["user_id", "item_id"], keep="first") \
            .reset_index(drop=True)
        df_imps = pd.merge(df.impressions.explode(), df[["user", "timestamp"]], left_index=True, right_index=True) \
            .rename(columns={"user": "user_id", "impressions": "item_id"}) \
            .drop_duplicates(["user_id", "item_id"], keep="first") \
            .reset_index(drop=True)

        # keep only users and items that clicked at least 5 users/items
        # so filter based on clicks, but also apply selection to impressions
        df_clicks = filter_interactions(df_clicks, 5, 5).reset_index(drop=True)
        df_imps = df_imps[df_imps.item_id.isin(df_clicks.item_id.unique())]
        df_imps = df_imps[df_imps.user_id.isin(df_clicks.user_id.unique())].reset_index(drop=True)

        # factorize users and items
        user_mapper = factorize_series_sorted(df_clicks["user_id"])
        item_mapper = factorize_series_sorted(df_clicks["item_id"])
        df_clicks["user_id"] = df_clicks.merge(user_mapper, left_on="user_id", right_index=True)["id"]
        df_clicks["item_id"] = df_clicks.merge(item_mapper, left_on="item_id", right_index=True)["id"]
        df_imps["user_id"] = df_imps.merge(user_mapper, left_on="user_id", right_index=True)["id"]
        df_imps["item_id"] = df_imps.merge(item_mapper, left_on="item_id", right_index=True)["id"]

        # load item metadata and apply id mapping
        item_metadata = pd.concat((
            pd.read_csv(base_path / "MINDlarge_train" / "news.tsv", sep="\t",
                        names=["item", "cat0", "cat1", "title", "description", "url", "data1", "data2"]),
            pd.read_csv(base_path / "MINDlarge_dev" / "news.tsv", sep="\t",
                        names=["item", "cat0", "cat1", "title", "description", "url", "data1", "data2"])
        )).drop_duplicates("item")
        item_metadata = item_metadata[item_metadata["item"].isin(item_mapper.index)]
        item_metadata["item_id"] = item_mapper.loc[item_metadata["item"]].values
        item_metadata = item_metadata.sort_values("item_id").reset_index(drop=True)

        with pd.HDFStore(h5_path) as store:
            store.put("interactions_clicks", df_clicks, format="table")
            store.put("interactions_imps", df_imps, format="table")
            store.put("item_metadata", item_metadata, format="table")

    with pd.HDFStore(h5_path, "r") as store:
        df_clicks = store.get("interactions_clicks")
        df_imps = store.get("interactions_imps")
        item_metadata = store.get("item_metadata")

    return df_clicks, df_imps, item_metadata
