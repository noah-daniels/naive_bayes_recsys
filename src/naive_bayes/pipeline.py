
from collections import defaultdict
import logging
import math
import sys
import time
import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy as sp

from naive_bayes.utils import get_top_ranks, mask_matrix_rowwise, sparse_reciprocal


class Scenario:
    def __init__(self, X_train, X_validation_in, X_validation_out, X_test_in, X_test_out):
        assert X_train.shape == X_validation_in.shape
        assert X_train.shape == X_validation_out.shape
        assert X_train.shape == X_test_in.shape
        assert X_train.shape == X_test_out.shape

        self.__train = X_train
        self.__validation_in = X_validation_in
        self.__validation_out = X_validation_out
        self.__test_in = X_test_in
        self.__test_out = X_test_out

    def get_validation_data(self):
        train = self.__train
        test_in = self.__validation_in
        test_out = self.__validation_out
        return train, test_in, test_out

    def get_full_data(self):
        train = self.__train + self.__validation_in + self.__validation_out
        test_in = self.__test_in
        test_out = self.__test_out
        return train, test_in, test_out

    @property
    def training_users(self):
        return np.nonzero(self.__train.sum(axis=1) > 0)[0]

    @property
    def validation_users(self):
        return np.nonzero((self.__validation_in + self.__validation_out).sum(axis=1) > 0)[0]

    @property
    def test_users(self):
        return np.nonzero((self.__test_in + self.__test_out).sum(axis=1) > 0)[0]


def strong_generalization(
    X,
    frac_users_train=0.8,
    frac_interactions_in=0.8,
    min_interactions_in=1,
    min_interactions_out=1,
    seed_users=None,
    seed_interactions=None,
):
    """
    Split an interactions matrix according to strong generalization.

    All users are divided into 3 sets:
    + `frac_users_train` users belong to either training or validation
        + `frac_users_train` of those belong to the **training** set
        + `1-`frac_users_train`` belong to the **validation** set
    + `1-frac_users_train` users belong to the **test** set

    The interactions of the validation and test users are split per user into fold-in and hold-out parts:
    + `frac_interactions_in` interactions belong to **fold-in** set
    + `frac_interactions_out` interactions belong to **hold-out** set

    Users who don't have at least `min_interactions_in` fold-in interactions and at least `min_interactions_out` hold-out interactions are removed.

    Validation experiments:
    + train using X_train
    + test using X_validation_in and X_validation_out

    Full experiments:
    + train using X_train + X_validation_in + X_validation_out
    + test using X_test_in and X_test_out
    """

    def split_users(X, frac_in, rng):
        # select random users
        users = np.nonzero(X.sum(axis=1) > 0)[0]
        user_idx = int(math.floor(len(users) * frac_in))
        rng.shuffle(users)
        users_train = users[:user_idx]
        users_test = users[user_idx:]

        # split matrix
        X_train = mask_matrix_rowwise(X, users_train)
        X_test = mask_matrix_rowwise(X, users_test)

        return X_train, X_test

    def split_interactions(X, frac_in, rng):
        user_count, item_count = X.shape

        X_in = X.copy()
        X_out = X.copy()
        X_in.resize(user_count, item_count + 1)
        X_out.resize(user_count, item_count + 1)

        for i in range(len(X.indptr) - 1):
            interactions = X.indices[X.indptr[i]: X.indptr[i + 1]].copy()
            interactions_idx = int(math.ceil(len(interactions) * frac_in))
            rng.shuffle(interactions)

            X_in.indices[X.indptr[i]: X.indptr[i + 1]] = item_count
            X_out.indices[X.indptr[i]: X.indptr[i + 1]] = item_count
            X_in.indices[X.indptr[i]: X.indptr[i + 1]][:interactions_idx] = interactions[:interactions_idx]
            X_out.indices[X.indptr[i]: X.indptr[i + 1]][interactions_idx:] = interactions[interactions_idx:]

        X_in.resize(X.shape)
        X_out.resize(X.shape)

        return X_in, X_out

    # split users
    # + first all users are divided into train+validation and test users
    # + train+validation users are divided into train and validation users
    rng_users = np.random.default_rng(seed_users)
    X_tv, X_test = split_users(X, frac_users_train, rng_users)
    X_train, X_validation = split_users(X_tv, frac_users_train, rng_users)
    del X_tv

    # split interactions
    # + test users are split into fold-in and hold-out interactions
    # + validation users are split into fold-in and hold-out interactions
    rng_interactions = np.random.default_rng(seed_interactions)
    X_test_in, X_test_out = split_interactions(X_test, frac_interactions_in, rng_interactions)
    X_validation_in, X_validation_out = split_interactions(X_validation, frac_interactions_in, rng_interactions)

    # ensure minimum number of in and out interactions
    valid_test_users = np.nonzero(
        (X_test_in.sum(axis=1) >= min_interactions_in) & (X_test_out.sum(axis=1) >= min_interactions_out)
    )[0]
    X_test_in = mask_matrix_rowwise(X_test_in, valid_test_users)
    X_test_out = mask_matrix_rowwise(X_test_out, valid_test_users)
    valid_validation_users = np.nonzero(
        (X_validation_in.sum(axis=1) >= min_interactions_in) & (X_validation_out.sum(axis=1) >= min_interactions_out)
    )[0]
    X_validation_in = mask_matrix_rowwise(X_validation_in, valid_validation_users)
    X_validation_out = mask_matrix_rowwise(X_validation_out, valid_validation_users)

    return Scenario(X_train, X_validation_in, X_validation_out, X_test_in, X_test_out)


def calculate_ndcg(y_true: sp.sparse.csr_array, y_ranks: sp.sparse.csr_array, k):
    hist_len = y_true.sum(axis=1).astype(np.int32)
    hist_len[hist_len > k] = k

    # precalculate idcg values up to K
    discount_template = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.r_[np.asarray([1]), np.cumsum(discount_template)]

    # calculate hits
    recos = y_ranks.copy()
    recos.data[y_ranks.data > k] = 0
    hits = recos.multiply(y_true)

    # calculate dcg per user
    hits.data = np.log2(hits.data + 1)
    dcg = y_true * sparse_reciprocal(hits)
    per_user_dcg = dcg.sum(axis=1)

    # calculate ndcg per user
    return (per_user_dcg / idcg[hist_len]).mean()


def calculate_recall(y_true: sp.sparse.csr_array, y_ranks: sp.sparse.csr_array, k):
    recos = y_ranks.copy()
    recos.data[recos.data > k] = 0
    hits = (recos.multiply(y_true) > 0).sum(axis=1)
    relevant = np.minimum(k, y_true.sum(axis=1))
    return np.mean(hits / relevant)


def calculate_precision(y_true: sp.sparse.csr_array, y_ranks: sp.sparse.csr_array, k):
    recos = y_ranks.copy()
    recos.data[recos.data > k] = 0
    hits = (recos.multiply(y_true) > 0).sum(axis=1)
    return np.mean(hits / k)


def process_scores(
    scores: sp.sparse.csr_array,
    blacklist: sp.sparse.csr_array | None = None,
    whitelist: sp.sparse.csr_array | None = None
):
    result = scores.copy()

    # remove items from blacklist
    if blacklist is not None:
        result -= result * blacklist

    # only keep items in whitelist
    if whitelist is not None:
        result *= whitelist

    return result


def run_algorithm(algorithm, parameter_generator, scenario: Scenario, validation=False, n_jobs=1):
    # check if all parameter combinations are valid
    for p in parameter_generator:
        algorithm(**p)

    # get split data
    X_train, X_test_in, X_test_out = scenario.get_validation_data() if validation else scenario.get_full_data()
    user_mask = X_test_out.sum(axis=1) > 0
    X_test_out = X_test_out[user_mask]

    def init_worker():
        """Configures logging inside the worker process."""
        logger = logging.getLogger("naive_bayes")
        logger.setLevel(logging.INFO)

        # Only add handler if it doesn't exist to prevent duplicates
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s (Process %(process)d): %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def run_single(p):
        logger = logging.getLogger("naive_bayes")

        res = {}
        start = time.time()

        # model fitting
        alg = algorithm(**p)
        alg.fit(X_train)
        end_fit = time.time()
        logger.info(f"Fitting for {alg.name} complete - Took {end_fit - start:.3}s")

        # model inference
        scores = alg.predict(X_test_in)
        end_inference = time.time()
        logger.info(f"Inference for {alg.name} complete - Took {end_inference - end_fit:.3}s")

        # transform scores into ranks
        scores = process_scores(scores, blacklist=X_test_in > 0)[user_mask]
        ranks = get_top_ranks(scores, 100)

        # calculate metrics
        for k in [5, 10, 25, 50, 100]:
            res[f"NDCG@{k}"] = calculate_ndcg(X_test_out, ranks, k)
            res[f"Recall@{k}"] = calculate_recall(X_test_out, ranks, k)
            res[f"Precision@{k}"] = calculate_precision(X_test_out, ranks, k)
        end_eval = time.time()
        logger.info(f"Evaluating {alg.name} complete - Took {end_eval - end_inference:.3}s")

        # save results
        res["identifier"] = alg.identifier
        res["fit_time"] = end_fit - start
        res["predict_time"] = end_inference - end_fit
        res["eval_time"] = end_eval - end_inference

        return res

    results = list(tqdm(joblib.Parallel(n_jobs, initializer=init_worker, max_nbytes=None, return_as="generator")(
        joblib.delayed(run_single)(p) for p in parameter_generator), total=len(parameter_generator), file=sys.stdout, smoothing=0))

    # results = defaultdict(list)
    # for p in tqdm(parameter_generator, file=sys.stdout):
    #     start = time.time()

    #     # model fitting
    #     alg = algorithm(**p)
    #     alg.fit(X_train)
    #     end_fit = time.time()
    #     logger.info(f"Fitting for {alg.name} complete - Took {end_fit - start:.3}s")

    #     # model inference
    #     scores = alg.predict(X_test_in)
    #     end_inference = time.time()
    #     logger.info(f"Inference for {alg.name} complete - Took {end_inference - end_fit:.3}s")

    #     # transform scores into ranks
    #     scores = process_scores(scores, blacklist=X_test_in > 0)[user_mask]
    #     ranks = get_top_ranks(scores, 100)

    #     # calculate metrics
    #     for k in [5, 10, 25, 50, 100]:
    #         results[f"NDCG@{k}"].append(calculate_ndcg(X_test_out, ranks, k))
    #         results[f"Recall@{k}"].append(calculate_recall(X_test_out, ranks, k))
    #         results[f"Precision@{k}"].append(calculate_precision(X_test_out, ranks, k))
    #     end_eval = time.time()
    #     logger.info(f"Evaluating {alg.name} complete - Took {end_eval - end_inference:.3}s")

    #     # save results
    #     results["identifier"].append(alg.identifier)
    #     results["fit_time"].append(end_fit - start)
    #     results["predict_time"].append(end_inference - end_fit)
    #     results["eval_time"].append(end_eval - end_inference)

    #     del alg

    # create dataframe
    return pd.DataFrame(results).set_index("identifier")
