import pandas as pd


def make_target_encoding(
    train: pd.DataFrame,
    cols: list[str],
    target: str,
    feature_name: str,
    smoothing: int = 50,
):
    """
    Compute a smoothed target encoding from the training data.
    """

    global_mean = train[target].mean()

    stats = (
        train.groupby(cols)[target]
        .agg(["mean", "count"])
        .reset_index()
    )

    stats[feature_name] = (
        stats["mean"] * stats["count"]
        + global_mean * smoothing
    ) / (stats["count"] + smoothing)

    return stats[cols + [feature_name]], global_mean


def apply_target_encoding(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    cols: list[str],
    feature_name: str,
    fill_value: float,
) -> pd.DataFrame:
    """
    Merge a target encoding into a dataframe.
    """

    df = df.merge(
        stats,
        on=cols,
        how="left",
    )

    df[feature_name] = df[feature_name].fillna(fill_value)

    return df
