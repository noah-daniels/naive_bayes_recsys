import pathlib

import pandas as pd

from naive_bayes.datasets.utils import BASE_PATH_RAW, filter_interactions, process_interactions_df


def load_ml100k(base_path: pathlib.Path | None = None, rating_threshold=4):
    """Avoid using because movieIds not consisten with other ml datasets"""

    if base_path is None:
        base_path = BASE_PATH_RAW / "ml" / "ml-100k"

    genres = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    df_movies = pd.read_table(
        base_path / "u.item",
        sep="|",
        encoding="latin1",
        names=[
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "url",
            *genres,
        ],
    )
    df_ratings = pd.read_csv(
        base_path / "u.data",
        sep="\t",
        encoding="latin1",
        names=["userId", "movieId", "rating", "timestamp"],
    )

    df_interactions = df_ratings[df_ratings["rating"] >= rating_threshold]
    df_interactions, user_mapper, item_mapper = process_interactions_df(
        df_interactions, "userId", "movieId"
    )

    item_metadata = (
        pd.merge(
            item_mapper,
            df_movies,
            left_index=True,
            right_on="movieId",
            how="left",
        )
        .rename(columns={"id": "item_id"})
        .set_index("item_id")
        .sort_index()
    )

    return df_interactions, item_metadata


def load_ml_general(df_ratings, df_movies, min_items_per_user, min_users_per_item, min_rating):
    df_interactions = df_ratings[df_ratings["rating"] >= min_rating].rename(
        columns={"userId": "user_id", "movieId": "item_id"})
    df_interactions = df_interactions.drop_duplicates(subset=["user_id", "item_id"])
    df_interactions = filter_interactions(df_interactions, min_items_per_user, min_users_per_item)
    df_interactions, _, item_mapper = process_interactions_df(df_interactions)

    item_metadata = (
        pd.merge(
            item_mapper,
            df_movies,
            left_index=True,
            right_on="movieId",
            how="left",
        )
        .rename(columns={"id": "item_id"})
        .set_index("item_id")
        .sort_index()
    )

    return df_interactions, item_metadata


def load_ml1m(base_path: pathlib.Path | None = None, min_rating=4):
    if base_path is None:
        base_path = BASE_PATH_RAW / "ml" / "ml-1m"

    df_ratings = pd.read_table(
        base_path / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    df_movies = pd.read_table(
        base_path / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "name", "genres"],
    )

    return load_ml_general(df_ratings, df_movies, min_items_per_user=5, min_users_per_item=5, min_rating=min_rating)


def load_ml25m(base_path: pathlib.Path | None = None, min_rating=4):
    if base_path is None:
        base_path = BASE_PATH_RAW / "ml" / "ml-25m"

    df_ratings = pd.read_csv(base_path / "ratings.csv")
    df_movies = pd.read_csv(base_path / "movies.csv")

    return load_ml_general(df_ratings, df_movies, min_items_per_user=5, min_users_per_item=5, min_rating=min_rating)
