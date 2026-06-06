import pandas as pd


def process_identifier(identifier):
    parameters = identifier.split('(')[1][:-1].split(',')
    algorithm = identifier.split('(')[0]
    parameter_dict = dict(map(lambda p: p.split('='), parameters))
    parameter_dict["identifier"] = identifier
    parameter_dict["algorithm"] = algorithm
    return parameter_dict


def add_params(df):
    param_df = pd.DataFrame(list(df["identifier"].map(process_identifier))).drop_duplicates()
    return pd.merge(df, param_df, left_on="identifier", right_on="identifier")


def filter_df(df, **kwargs):
    mask = True
    for k, v in kwargs.items():
        mask &= df[k] == v
    return df[mask]


def get_best_params(df, metric):
    params = {}
    values = df.sort_values(metric).iloc[-1]
    for col in df.columns:
        if col.startswith("NDCG") or col.startswith("Recall") or col.startswith("Precision") or col.endswith("time") or col in ["identifier", "dataset", "algorithm"]:
            continue
        else:
            if col == "max_k" and values[col] == 1000000:
                params[col] = None
            else:
                params[col] = values[col]
    return params


def load_results(path, include_pruning_results):
    df_ease = add_params(pd.concat((
        pd.read_csv(path / "amazon_books_ease.csv").assign(dataset="Amazon Books"),
        pd.read_csv(path / "mind_ease.csv").assign(dataset="MIND"),
        pd.read_csv(path / "ml25m_ease.csv").assign(dataset="ML25M"),
    )))
    df_ease["l2"] = df_ease["l2"].astype(float).round(decimals=4)

    df_knn = add_params(pd.concat((
        pd.read_csv(path / "amazon_books_knn.csv").assign(dataset="Amazon Books"),
        pd.read_csv(path / "mind_knn.csv").assign(dataset="MIND"),
        pd.read_csv(path / "ml25m_knn.csv").assign(dataset="ML25M"),
    )))
    df_knn["max_k"] = df_knn["max_k"].replace("None", "1000000").astype(int)
    df_knn["bias_smoothing"] = df_knn["bias_smoothing"].astype(float).round(decimals=5)
    df_knn["bias_weight"] = df_knn["bias_weight"].astype(float).round(decimals=5)

    df_bernoulli = pd.concat((
        pd.read_csv(path / "amazon_books_bernoulli.csv").assign(dataset="Amazon Books"),
        pd.read_csv(path / "mind_bernoulli.csv").assign(dataset="MIND"),
        pd.read_csv(path / "ml25m_bernoulli.csv").assign(dataset="ML25M"),
    ))
    if include_pruning_results:
        df_bernoulli = pd.concat((
            df_bernoulli,
            pd.read_csv(path / "amazon_books_bernoulli_pruning.csv").assign(dataset="Amazon Books"),
            pd.read_csv(path / "mind_bernoulli_pruning.csv").assign(dataset="MIND"),
            pd.read_csv(path / "ml25m_bernoulli_pruning.csv").assign(dataset="ML25M"),
        ))
    df_bernoulli = add_params(df_bernoulli)
    df_bernoulli["max_k"] = df_bernoulli["max_k"].replace("None", "1000000").astype(int)
    df_bernoulli["bias_weight"] = df_bernoulli["bias_weight"].astype(float).round(decimals=5)
    df_bernoulli["bias_smoothing"] = df_bernoulli["bias_smoothing"].astype(float).round(decimals=5)
    df_bernoulli["weights_smoothing_strength"] = df_bernoulli["weights_smoothing_strength"].astype(
        float).round(decimals=5)
    df_bernoulli["weights_complement_weight"] = df_bernoulli["weights_complement_weight"].astype(
        float).round(decimals=5)
    df_bernoulli["model_complement_weight"] = df_bernoulli["model_complement_weight"].astype(float).round(decimals=5)

    df_multinomial = pd.concat((
        pd.read_csv(path / "amazon_books_multinomial.csv").assign(dataset="Amazon Books"),
        pd.read_csv(path / "mind_multinomial.csv").assign(dataset="MIND"),
        pd.read_csv(path / "ml25m_multinomial.csv").assign(dataset="ML25M"),

    ))
    if include_pruning_results:
        df_multinomial = pd.concat((
            df_multinomial,
            pd.read_csv(path / "amazon_books_multinomial_pruning.csv").assign(dataset="Amazon Books"),
            pd.read_csv(path / "mind_multinomial_pruning.csv").assign(dataset="MIND"),
            pd.read_csv(path / "ml25m_multinomial_pruning.csv").assign(dataset="ML25M"),
        ))
    df_multinomial = add_params(df_multinomial)
    df_multinomial["max_k"] = df_multinomial["max_k"].replace("None", "1000000").astype(int)
    df_multinomial["centering"] = df_multinomial["centering"].replace("False", "").astype(bool)
    df_multinomial["absolute_pruning"] = df_multinomial["absolute_pruning"].replace("False", "").astype(bool)
    df_multinomial["bias_smoothing"] = df_multinomial["bias_smoothing"].astype(float).round(decimals=5)
    df_multinomial["bias_weight"] = df_multinomial["bias_weight"].astype(float).round(decimals=5)
    df_multinomial["weights_smoothing_strength"] = df_multinomial["weights_smoothing_strength"].astype(
        float).round(decimals=5)

    return df_ease, df_knn, df_bernoulli, df_multinomial
